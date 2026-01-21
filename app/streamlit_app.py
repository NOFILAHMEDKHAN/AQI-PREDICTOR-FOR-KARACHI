import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import os
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AQI PREDICTOR FOR KARACHI",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded"
)
load_dotenv()

# --- 2. PROFESSIONAL STYLING (CSS) ---
st.markdown("""
<style>
    .big-font { font-size: 18px !important; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #0E1117; }
    .stAlert { padding: 10px; border-radius: 5px; }
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND: HOPSWORKS CONNECTION ---
@st.cache_resource
def init_hopsworks(key):
    try:
        project = hopsworks.login(api_key_value=key)
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        # Connect to Feature Group (Version 3)
        try:
            fg = fs.get_feature_group(name="karachi_aqi_pro", version=3)
        except:
            st.warning("⚠️ Using Fallback Data (Version 1). Run feature_pipeline.py v3 for best results.")
            fg = fs.get_feature_group(name="karachi_aqi_pro", version=1)
        
        # Load ALL Models for Comparison
        models = []
        target_models = ["aqi_gb_pro", "aqi_rf_pro", "aqi_ridge_pro"]
        
        for name in target_models:
            try:
                # 👇 Force Latest Version
                models_list = mr.get_models(name)
                remote = max(models_list, key=lambda x: x.version)
                
                path = remote.download()
                model = joblib.load(path + f"/{name}.pkl")
                
                # Extract Metrics
                metrics = remote.training_metrics
                r2 = metrics.get("r2", 0.0)
                rmse = metrics.get("rmse", 0.0)
                mae = metrics.get("mae", 0.0)
                
                # Friendly Labels
                if "gb" in name: label, type_ = "XGBoost (Gradient Boosting)", "Tree"
                elif "rf" in name: label, type_ = "Random Forest", "Tree"
                else: label, type_ = "Ridge Regression", "Linear"
                
                models.append({
                    "name": name, 
                    "label": label, 
                    "model": model, 
                    "r2": float(r2), 
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "type": type_
                })
            except Exception as e: 
                print(f"Skipping {name}: {e}")
            
        return fg, models
    except Exception as e:
        return None, None

# --- 4. WEATHER API (LIVE FORECAST) ---
@st.cache_data(ttl=3600) 
def get_weather_forecast():
    """Fetches 7-day forecast from Open-Meteo"""
    lat, lon = 24.86, 67.01
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m&forecast_days=7"
    try:
        response = requests.get(url).json()
        hourly = response['hourly']
        df = pd.DataFrame({
            'date': pd.to_datetime(hourly['time']),
            'temperature': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'rain': hourly['precipitation'],
            'wind_speed': hourly['wind_speed_10m'],
            'wind_dir': hourly['wind_direction_10m'] # Needed for vectors
        })
        df['date'] = df['date'].dt.tz_localize(None)
        df['hour'] = df['date'].dt.hour
        return df
    except:
        return pd.DataFrame()

# --- 5. PREDICTION ENGINE (RECURSIVE) ---
def generate_forecast(model, history_df, weather_df, hours=72):
    # Setup Buffer (Last 48h of history)
    history_buffer = history_df.sort_values("date").tail(48).copy()
    history_buffer = history_buffer.set_index("date")
    
    future_preds = []
    
    # Start predicting immediately after the last known history point
    last_hist_time = history_df['date'].max()
    future_weather = weather_df[weather_df['date'] > last_hist_time].head(hours)
    
    if future_weather.empty: return pd.DataFrame()

    for i, row in future_weather.iterrows():
        target_time = row['date']
        
        # A. Calculate Time Lags (Recursive Step)
        def get_aqi_at(t):
            if t in history_buffer.index: return history_buffer.loc[t, 'aqi']
            return history_buffer['aqi'].iloc[-1]
            
        lag_1 = get_aqi_at(target_time - timedelta(hours=1))
        lag_24 = get_aqi_at(target_time - timedelta(hours=24))
        
        # B. Smart Feature Engineering (Replicating Pipeline Logic)
        rads = np.deg2rad(row['wind_dir'])
        wind_u = row['wind_speed'] * np.cos(rads)
        wind_v = row['wind_speed'] * np.sin(rads)
        h = row['hour']
        is_rush = 1 if (8<=h<=10) or (17<=h<=20) else 0
        
        # C. Prepare Input Vector
        input_data = pd.DataFrame([{
            'aqi_lag_1': lag_1, 
            'aqi_lag_24': lag_24,
            'temperature': row['temperature'], 
            'humidity': row['humidity'],
            'rain': row['rain'],
            'wind_u': wind_u,
            'wind_v': wind_v,
            'is_rush_hour': is_rush
        }])
        
        # D. Predict
        try:
            pred_aqi = model.predict(input_data)[0]
        except:
            # Fallback for older models
            alt_input = input_data.copy()
            alt_input['wind_speed'] = row['wind_speed']
            alt_input['hour'] = h
            cols_v1 = ['aqi_lag_1', 'aqi_lag_24', 'temperature', 'humidity', 'wind_speed', 'hour']
            pred_aqi = model.predict(alt_input[cols_v1])[0]

        # E. Store & Update Buffer
        future_preds.append({'date': target_time, 'aqi': pred_aqi})
        new_row = pd.DataFrame({'aqi': [pred_aqi]}, index=[target_time])
        history_buffer = pd.concat([history_buffer, new_row])
        
    return pd.DataFrame(future_preds)

# --- UI START ---
key = os.getenv("HOPSWORKS_API_KEY")
if not key: key = st.sidebar.text_input("Hopsworks API Key", type="password")
if not key: st.stop()

with st.spinner("🚀 Booting AQI Command Center..."):
    fg, models = init_hopsworks(key)
    if not fg or not models: st.error("System Offline. Check API Key."); st.stop()

# Select Best Model automatically based on R2 Score
models.sort(key=lambda x: x['r2'], reverse=True)
best_model = models[0]

# --- LOAD AND RENAME DATA ---
raw_data = fg.select_all().read()

# 1. Convert timestamp and rename to 'date'
if 'timestamp' in raw_data.columns:
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], unit='ms')
    raw_data = raw_data.rename(columns={'timestamp': 'date'})
else:
    raw_data['date'] = pd.to_datetime(raw_data['date'])

# 2. Sort
raw_data = raw_data.sort_values('date')

current_time = pd.Timestamp.now().replace(microsecond=0)
real_history = raw_data[raw_data['date'] <= current_time].sort_values("date")
weather_forecast = get_weather_forecast()

# --- DASHBOARD HEADER ---
st.title("🏭 Karachi AQI Command Center")
st.markdown(f"**Live Status:** {current_time.strftime('%A, %d %B %Y | %H:%M')} &nbsp; | &nbsp; **Active Engine:** `{best_model['label']}`")

# --- NAVIGATION TABS ---
tab_forecast, tab_eval, tab_analysis = st.tabs(["📡 Live Forecast (72h)", "📊 Model Evaluation", "🧠 Feature Intelligence (SHAP)"])

# =========================================================
# TAB 1: LIVE FORECAST (THE MAIN DASHBOARD)
# =========================================================
with tab_forecast:
    if not real_history.empty:
        latest = real_history.iloc[-1]
        aqi_now = int(latest['aqi'])
        
        # Color Logic
        if aqi_now <= 50: status, color = "Good", "normal"
        elif aqi_now <= 100: status, color = "Moderate", "off"
        elif aqi_now <= 150: status, color = "Unhealthy", "inverse"
        else: status, color = "Hazardous", "inverse"
        
        # Wind Fix
        if 'wind_speed' in latest:
            wind_disp = latest['wind_speed']
        elif 'wind_u' in latest and 'wind_v' in latest:
            wind_disp = np.sqrt(latest['wind_u']**2 + latest['wind_v']**2)
        else:
            wind_disp = 0.0
        
        # --- CARDS ROW ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current AQI", aqi_now, status, delta_color=color)
        c2.metric("Temperature", f"{latest['temperature']} °C")
        c3.metric("Wind Condition", f"{wind_disp:.1f} km/h")
        rain_val = latest['rain'] if 'rain' in latest else 0
        c4.metric("Precipitation", f"{rain_val} mm")

        # --- [NEW] GAUGE CHART (VISUAL UPGRADE) ---
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = aqi_now,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Live Hazard Level", 'font': {'size': 24}},
            delta = {'reference': 50, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color.replace("inverse", "darkred").replace("normal", "green").replace("off", "orange")},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': "#00e400"},
                    {'range': [50, 100], 'color': "#ffff00"},
                    {'range': [100, 150], 'color': "#ff7e00"},
                    {'range': [150, 200], 'color': "#ff0000"},
                    {'range': [200, 300], 'color': "#7e0023"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 150}}))
        
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # 2. MAIN 3-DAY FORECAST CHART
    st.subheader("🔮 Predictive Horizon (Next 3 Days)")
    
    forecast_df = generate_forecast(best_model['model'], real_history, weather_forecast, hours=72)
    
    if not forecast_df.empty:
        hist_data = real_history.tail(48)[['date', 'aqi']].assign(Type='History')
        fore_data = forecast_df[['date', 'aqi']].assign(Type='Forecast')
        combined = pd.DataFrame(hist_data.to_dict('records') + fore_data.to_dict('records'))
        
        fig = px.line(combined, x='date', y='aqi', color='Type',
                      color_discrete_map={"History": "gray", "Forecast": "#00CC96"},
                      title="Real-Time AQI Trajectory")
        
        fig.add_hline(y=150, line_dash="dot", annotation_text="Unhealthy Threshold", line_color="red")
        fig.update_layout(hovermode="x unified", xaxis_title="Time", yaxis_title="AQI Level")
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. DAILY SUMMARY CARDS
        st.caption("Daily Average Forecasts:")
        forecast_df['day_name'] = forecast_df['date'].dt.day_name()
        daily_avgs = forecast_df.groupby('day_name', sort=False)['aqi'].mean()
        
        cols = st.columns(3)
        for i, (day, val) in enumerate(daily_avgs.items()):
            if i < 3:
                with cols[i]:
                    val_int = int(val)
                    if val_int <= 50: class_label = "Good 🌱"
                    elif val_int <= 100: class_label = "Moderate 😐"
                    elif val_int <= 150: class_label = "Unhealthy 😷"
                    else: class_label = "Hazardous ☠️"
                        
                    st.info(f"**{day}**")
                    st.metric("Avg Predicted AQI", f"{val_int}", delta=class_label, delta_color="off")

# =========================================================
# TAB 2: MODEL EVALUATION (FIXED LOGIC)
# =========================================================
with tab_eval:
    st.header("⚔️ Model Battle Arena")
    st.markdown("We trained **three advanced algorithms** to find the most accurate predictor. Here are the results from the test set.")
    
    # 1. METRICS TABLE
    eval_df = pd.DataFrame(models)[['label', 'r2', 'rmse', 'mae']]
    eval_df.columns = ['Model Name', 'Accuracy (R²)', 'Error Peaks (RMSE)', 'Avg Error (MAE)']
    
    # Highlight the winner (Green)
    st.dataframe(eval_df.style.highlight_max(axis=0, subset=['Accuracy (R²)'], color='#d4edda')
                 .highlight_min(axis=0, subset=['Error Peaks (RMSE)', 'Avg Error (MAE)'], color='#d4edda'),
                 use_container_width=True)
    
    col_chart, col_txt = st.columns([2, 1])
    
    with col_chart:
        melted_df = eval_df.melt(id_vars="Model Name", var_name="Metric", value_name="Score")
        fig_perf = px.bar(melted_df, x="Model Name", y="Score", color="Metric", barmode="group",
                          title="Performance Metrics Comparison", text_auto=".2f")
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_txt:
        # 3. WINNER EXPLANATION (SCIENTIFICALLY ROBUST)
        # Find winners for all 3 metrics
        r2_winner = eval_df.sort_values("Accuracy (R²)", ascending=False).iloc[0]
        mae_winner = eval_df.sort_values("Avg Error (MAE)", ascending=True).iloc[0]
        rmse_winner = eval_df.sort_values("Error Peaks (RMSE)", ascending=True).iloc[0]
        
        # The official winner is whoever is at the top of the sorted list (Best R2)
        final_winner = r2_winner 
        
        st.success(f"🏆 **Winner: {final_winner['Model Name']}**")
        
        # --- DYNAMIC EXPLANATION GENERATOR ---
        reasoning = f"**Why we chose this model:**\n\n"
        
        # 1. R2 Logic
        reasoning += f"* It demonstrates high **Variance Explanation (R²: {final_winner['Accuracy (R²)']:.2%})**.\n"

        # 2. The "Tie-Breaker" Logic (MAE vs RMSE)
        if final_winner['Model Name'] == mae_winner['Model Name']:
             # Easy Case: It won everything
             reasoning += f"* It is superior across all metrics, including lowest **Average Error ({final_winner['Avg Error (MAE)']:.2f})**.\n"
        else:
             # Hard Case: It lost on MAE but won on RMSE/R2
             mae_diff = final_winner['Avg Error (MAE)'] - mae_winner['Avg Error (MAE)']
             reasoning += f"* While {mae_winner['Model Name']} had slightly lower average error (Diff: {mae_diff:.2f}), it failed to catch the peaks.\n"
             reasoning += f"* **{final_winner['Model Name']}** achieved the lowest **RMSE ({final_winner['Error Peaks (RMSE)']:.2f})**.\n"
             reasoning += "* **Why this matters:** RMSE penalizes large errors more heavily. In AQI prediction, avoiding large errors (missing a hazardous spike) is more critical than a slightly better average."

        st.markdown(reasoning)

# =========================================================
# TAB 3: ANALYSIS (WITH SHAP + WATERFALL UPGRADE)
# =========================================================
with tab_analysis:
    st.header("🧠 X-Ray: Why does the model predict this?")
    st.markdown("This analysis uses **SHAP (SHapley Additive exPlanations)**, a Nobel-prize winning Game Theory approach to explain exactly how each feature pushed the AQI up or down.")

    # We need Training Data to initialize SHAP
    # For speed, we just grab a sample of the recent history used for prediction
    X_sample = real_history.tail(100).copy()
    
    # Drop non-feature columns if they exist in history
    feature_cols = ['aqi_lag_1', 'aqi_lag_24', 'temperature', 'humidity', 'rain', 'wind_u', 'wind_v', 'is_rush_hour']
    # Ensure only valid columns are kept
    valid_cols = [c for c in feature_cols if c in X_sample.columns]
    X_sample = X_sample[valid_cols].fillna(0) # Safety fill

    if st.button("🚀 Calculate SHAP Values (Takes 10s)"):
        with st.spinner("Calculating SHAP values..."):
            try:
                # 1. Initialize Explainer (Modern API)
                # TreeExplainer is fast for XGBoost/RandomForest
                explainer = shap.TreeExplainer(best_model['model'])
                
                # 2. Calculate SHAP values (Returns Explanation Object)
                shap_values = explainer(X_sample)
                
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Global Feature Importance")
                    st.caption("Overall, what drives pollution in Karachi?")
                    # Beeswarm Plot (Modern Summary)
                    fig, ax = plt.subplots()
                    shap.plots.beeswarm(shap_values, show=False)
                    st.pyplot(fig)
                    
                with c2:
                    st.subheader("Feature Impact Direction")
                    st.caption("Bar chart of mean importance")
                    # Bar Plot (Simple Magnitude)
                    fig2, ax2 = plt.subplots()
                    shap.plots.bar(shap_values, show=False)
                    st.pyplot(fig2)

                st.markdown("---")
                st.subheader("📍 Local Explanation (Why is AQI high/low RIGHT NOW?)")
                st.caption("This 'Waterfall' shows exactly how today's weather pushed the prediction up or down from the average.")
                
                # WATERFALL PLOT (The Upgrade)
                # We take the very last row (Current Hour)
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_values[-1], show=False)
                st.pyplot(fig3)
                    
            except Exception as e:
                st.warning(f"Could not calculate SHAP for this model type ({best_model['type']}). Falling back to native importance.")
                st.code(str(e))
                # Fallback to your original bar chart if SHAP fails (e.g. for Ridge)
                if hasattr(best_model['model'], 'feature_importances_'):
                    imp = best_model['model'].feature_importances_
                    imp_df = pd.DataFrame({'Feature': valid_cols, 'Importance': imp}).sort_values('Importance', ascending=True)
                    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                     color='Importance', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Click the button above to run the deep SHAP analysis.")
import os
import time
import joblib
import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")

def train_smart_models():
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # --- 1. CONFIGURATION (TARGETING VERSION 3) ---
    fg_name = "karachi_aqi_pro"
    fg_version = 3  # <--- CRITICAL: Version 3 has the new Smart Features
    
    # Unique ID to ensure fresh Training Data every time
    unique_id = int(time.time())
    fv_name = f"karachi_aqi_view_v{unique_id}" 
    fv_version = 1

    print(f"🆔 Generated Unique View Name: {fv_name}")

    # --- 2. GET FEATURE GROUP ---
    try:
        fg = fs.get_feature_group(fg_name, fg_version)
    except:
        print(f"❌ ERROR: Feature Group '{fg_name}' (v{fg_version}) not found. Run feature_pipeline.py first.")
        return

    # --- 3. CREATE FEATURE VIEW ---
    # Selecting High-Impact Features (Rain, Wind Vectors, Rush Hour)
    print("🧠 Selecting High-Impact Features...")
    selected_features = [
        "aqi_lag_1", 
        "aqi_lag_24", 
        "temperature", 
        "humidity", 
        "rain",            # New: Washout Effect
        "wind_u",          # New: Wind Vector X
        "wind_v",          # New: Wind Vector Y
        "is_rush_hour"     # New: Traffic Logic
    ]
    label = ["aqi"]
    
    query = fg.select(selected_features + label)

    try:
        feature_view = fs.create_feature_view(
            name=fv_name,
            version=fv_version,
            query=query,
            labels=label,
            description=f"Smart View with Pro Metrics (v{unique_id})"
        )
    except Exception as e:
        print(f"❌ CRITICAL ERROR creating view: {e}")
        return

    # --- 4. CREATE & LOAD TRAINING DATA ---
    print("📊 Creating Training Dataset (this may take 1-2 minutes)...")
    
    version, job = feature_view.create_training_data(
        description="Training Dataset v3",
        data_format="csv",
        write_options={"wait_for_job": True}
    )
    
    print(f"✅ Training Dataset (v{version}) ready. Retrieving...")
    X, y = feature_view.get_training_data(version)
    
    # Simple sorting by index/time usually helps split recent data for testing
    # But random split is standard for general accuracy checks
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Data Shape - Train: {X_train.shape} | Test: {X_test.shape}")

    # --- 5. DEFINE MODELS ---
    models = {
        # Gradient Boosting: Best for complex non-linear patterns (Smog spikes)
        "aqi_gb_pro": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1),
        
        # Random Forest: Robust and stable
        "aqi_rf_pro": RandomForestRegressor(n_estimators=150, max_depth=12),
        
        # Ridge: Baseline to ensure we are beating linear logic
        "aqi_ridge_pro": Ridge(alpha=1.0)
    }

    os.makedirs("models", exist_ok=True)
    
    # --- 6. TRAINING & EVALUATION LOOP ---
    print("\n⚔️  Training & Evaluating Models...")
    
    for name, model in models.items():
        print(f"\n   ...Training {name}")
        model.fit(X_train, y_train.values.ravel()) 

        # --- A. NUMERICAL PREDICTION ---
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds) # Human Readable Error
        
        print(f"   👉 {name} Metrics:")
        print(f"      R2 (Accuracy): {r2:.3f}")
        print(f"      RMSE (Peaks):  {rmse:.2f}")
        print(f"      MAE (Avg Err): {mae:.2f}")

        # --- B. CATEGORICAL ALERT ACCURACY (The "Pro" Check) ---
        # Convert Numbers to AQI Categories for safety check
        # Bins: 0-50 (Good), 50-100 (Moderate), 100-150 (Unhealthy), 150+ (Hazardous)
        bins = [0, 50, 100, 150, 1000]
        labels = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']
        
        # We use .values.ravel() to handle potential dataframe/series shapes
        y_test_cat = pd.cut(y_test.values.ravel(), bins=bins, labels=labels)
        preds_cat = pd.cut(preds, bins=bins, labels=labels)
        
        print(f"      🚦 Alert Classification Report:")
        # We print a simplified report to the console
        cls_report = classification_report(y_test_cat, preds_cat, labels=labels, zero_division=0)
        print("\n".join(["      " + line for line in cls_report.split("\n")]))

        # --- C. SAVE & REGISTER ---
        path = f"models/{name}.pkl"
        joblib.dump(model, path)
        
        input_example = X_train.sample(1)
        
        mr_model = mr.python.create_model(
            name=name,
            # We now save ALL 3 metrics to the registry
            metrics={"r2": r2, "rmse": rmse, "mae": mae},
            description=f"Smart Model v3 (w/ Rain & Vectors)",
            input_example=input_example
        )
        mr_model.save(path)
        print(f"      📤 Registered {name} to Hopsworks")

    print("\n✅ SUCCESS! All models trained, evaluated, and pushed.")

if __name__ == "__main__":
    train_smart_models()
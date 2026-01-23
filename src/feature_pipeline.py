import os
import hopsworks
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")

def fetch_impactful_data():
    print("🌍 API: Fetching High-Impact Data (Last 90 Days + Forecast)...")
    LAT, LON = 24.86, 67.01
    w_metrics = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m"
    w_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={w_metrics}&past_days=90"
    
    try:
        w_resp = requests.get(w_url).json()
        if 'error' in w_resp: raise Exception(w_resp)
        w_hourly = w_resp['hourly']
    except Exception as e:
        print(f"❌ Weather API Failed: {e}")
        raise e
    
    df_weather = pd.DataFrame({
        'timestamp': pd.to_datetime(w_hourly['time']),
        'temperature': w_hourly['temperature_2m'],
        'humidity': w_hourly['relative_humidity_2m'],
        'rain': w_hourly['precipitation'],
        'wind_speed': w_hourly['wind_speed_10m'],
        'wind_dir': w_hourly['wind_direction_10m']
    })

    p_metrics = "pm10,pm2_5"
    p_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&hourly={p_metrics}&past_days=90"
    
    try:
        p_resp = requests.get(p_url).json()
        if 'error' in p_resp: raise Exception(p_resp)
        p_hourly = p_resp['hourly']
    except Exception as e:
        print(f"❌ Pollutant API Failed: {e}")
        raise e
    
    df_pollutants = pd.DataFrame({
        'timestamp': pd.to_datetime(p_hourly['time']),
        'pm2_5': p_hourly['pm2_5'],
        'pm10': p_hourly['pm10']
    })
    
    return pd.merge(df_weather, df_pollutants, on='timestamp', how='inner')

def engineer_features(df):
    print("🛠️  Engineering High-Impact Features...")
    
    # 1. AQI -> int32 (Matches your Schema)
    df['aqi'] = df['pm2_5'].apply(lambda x: x * 3.8 if x > 0 else 0).astype('int32')
    
    # 2. Wind Vectors
    rads = np.deg2rad(df['wind_dir'])
    df['wind_u'] = df['wind_speed'] * np.cos(rads)
    df['wind_v'] = df['wind_speed'] * np.sin(rads)
    
    # 3. Time
    df['hour'] = df['timestamp'].dt.hour
    
    # 4. Rush Hour -> int64 (BigInt - FIXING YOUR ERROR HERE)
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if (8<=h<=10) or (17<=h<=20) else 0).astype('int64')
    
    # 5. Lags
    df = df.sort_values("timestamp")
    df['aqi_lag_1'] = df['aqi'].shift(1)
    df['aqi_lag_24'] = df['aqi'].shift(24)
    df = df.dropna()
    
    # 6. Humidity -> int64 (BigInt - Matches Schema)
    df['humidity'] = df['humidity'].round().astype('int64')
    
    # 7. Timestamp -> int64
    df['timestamp'] = df['timestamp'].astype('int64') // 10**6
    
    cols_to_keep = [
        'timestamp', 'aqi', 'temperature', 'humidity', 'rain', 
        'wind_u', 'wind_v', 'is_rush_hour', 'aqi_lag_1', 'aqi_lag_24'
    ]
    return df[cols_to_keep]

def main():
    try:
        raw_data = fetch_impactful_data()
        clean_data = engineer_features(raw_data)
        
        print("☁️  Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=API_KEY)
        fs = project.get_feature_store()
        
        fg_name = "karachi_aqi_pro"
        version = 3
        
        print(f"📤 Uploading processed data to {fg_name} (v{version})...")
        aqi_fg = fs.get_or_create_feature_group(
            name=fg_name,
            version=version,
            primary_key=["timestamp"],
            event_time="timestamp",
            description="High-Impact AQI Data",
            online_enabled=True
        )
        # The "wait_for_job=False" part tells Python:
        # "Upload the data and finish immediately. Don't wait for the server."
        aqi_fg.insert(clean_data, write_options={"wait_for_job": False})
        
        print("✅ Success! Data is stored and ready.")
    except Exception as e:
        print(f"🚨 Pipeline Failed: {e}")

if __name__ == "__main__":
    main()
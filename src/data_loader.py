import requests
import pandas as pd
from datetime import datetime, timedelta

def get_karachi_weather_pollution(days_back=7):
    """
    Fetches hourly data for Karachi (Lat: 24.86, Lon: 67.01).
    
    Args:
        days_back (int): 
            - Use 120 for EDA/Backfill (History).
            - Use 3 for Hourly Automation (Speed).
    """
    print(f"🌍 API Call: Fetching last {days_back} days for Karachi...")
    LAT, LON = 24.86, 67.01
    
    # Calculate Dynamic Dates
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d") # +3 days forecast
    
    # 1. Open-Meteo API Endpoints
    aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&hourly=pm2_5,pm10&start_date={start}&end_date={end}"
    w_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&start_date={start}&end_date={end}"
    
    try:
        # 2. Fetch Data
        aq_response = requests.get(aq_url)
        w_response = requests.get(w_url)

        # Check for HTTP errors (e.g., 404, 500)
        aq_response.raise_for_status()
        w_response.raise_for_status()

        aq_res = aq_response.json()
        w_res = w_response.json()
        
        # 3. Convert to Pandas
        df_aq = pd.DataFrame(aq_res['hourly'])
        df_w = pd.DataFrame(w_res['hourly'])
        
        # 4. Merge Weather into Pollution Data (Safer than direct assignment)
        # This ensures the 'time' columns match perfectly before combining
        df = pd.merge(df_aq, df_w, on='time', how='inner')

        # 5. Clean up Timestamp
        df['time'] = pd.to_datetime(df['time'])
        
        # Optional: Rename columns for clarity if needed
        df = df.rename(columns={
            'temperature_2m': 'temperature',
            'relative_humidity_2m': 'humidity',
            'wind_speed_10m': 'wind_speed'
        })
        
        return df
        
    except Exception as e:
        print(f"❌ Critical API Error: {e}")
        return pd.DataFrame()

# ==========================================
# EXECUTION BLOCK (Modified for 4 Months)
# ==========================================
# ==========================================
# EXECUTION BLOCK (Modified for 90 Days)
# ==========================================
if __name__ == "__main__":
    # CHANGED: 120 -> 90 (The maximum allowed by the API)
    df = get_karachi_weather_pollution(days_back=90)
    
    if not df.empty:
        print("\n✅ Data Successfully Loaded!")
        print(f"Shape: {df.shape}")  # Expect roughly (2200+, 6)
        print("\nFirst 5 Rows:")
        print(df.head())
        print("\nLast 5 Rows (Check if forecast is included):")
        print(df.tail())
    else:
        print("\n⚠️ Returned empty dataframe.")
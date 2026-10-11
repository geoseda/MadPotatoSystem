import os
import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

CSV_FILE = 'sales_data.csv'
MODEL_OUTPUT = 'prophet_model.joblib'
FORECAST_OUTPUT = 'prophet_forecast.joblib'

def train_and_save_prophet():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found. Ensure dataset is in the directory.")
        return

    # 1. Load Data with flexible date parsing
    df = pd.read_csv(CSV_FILE)
    df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
    df = df.dropna().sort_values('ds').reset_index(drop=True)
    
    print(f"📊 Dataset Loaded: {len(df)} historical entries from {df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}")

    # 2. Smooth daily noise using a 7-day rolling window to capture true demand baseline
    df['y_raw'] = df['y']
    df['y'] = df['y_raw'].rolling(window=7, min_periods=1, center=True).mean()

    # 3. Configure & Fit Meta Prophet Model
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.08
    )
    model.fit(df)

    # 4. In-Sample Performance Evaluation
    in_sample = model.predict(df[['ds']])
    mae = mean_absolute_error(df['y'], in_sample['yhat'])
    rmse = np.sqrt(mean_squared_error(df['y'], in_sample['yhat']))
    mape = np.mean(np.abs((df['y'] - in_sample['yhat']) / df['y'])) * 100

    print("\n=== 📈 Meta Prophet Training Evaluation ===")
    print(f"Mean Absolute Error (MAE)     : ₱{mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ₱{rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # 5. Generate 90-Day Future Forecast Dataframe
    future = model.make_future_dataframe(periods=90, freq='D')
    forecast = model.predict(future)
    forecast['ds_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')

    # Assign Performance Tiers
    def assign_performance_tier(yhat):
        if yhat < 2000:
            return "Operating Deficit 🚨 (<₱2,000)"
        elif yhat < 3000:
            return "Underperforming ⚠️ (₱2,000 - ₱2,999)"
        elif yhat <= 4800:
            return "Acceptable / Typical Day ✅ (₱3,800 - ₱4,800)"
        else:
            return "High-Performing 🚀 (>₱5,000)"

    forecast['performance'] = forecast['yhat'].apply(assign_performance_tier)

    # 6. Save Model and Forecast Cache to Disk
    joblib.dump(model, MODEL_OUTPUT)
    joblib.dump(forecast, FORECAST_OUTPUT)
    print(f"\n✅ Saved Optimized Meta Prophet Model -> `{MODEL_OUTPUT}`")
    print(f"✅ Saved 90-Day Forecast Cache -> `{FORECAST_OUTPUT}`")

if __name__ == '__main__':
    train_and_save_prophet()
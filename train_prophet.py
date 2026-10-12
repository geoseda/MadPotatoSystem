import os
import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

CSV_FILE = 'sales_data.csv'
MODEL_OUTPUT = 'prophet_model.joblib'
FORECAST_OUTPUT = 'prophet_forecast.joblib'

def add_custom_features(df):
    """Adds Payday and Weekend binary indicators to capture store surges."""
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    
    # Philippine Payday window (14th-16th and 29th-31st)
    df['is_payday'] = df['day'].apply(lambda d: 1 if (14 <= d <= 16) or (29 <= d <= 31) else 0)
    # Weekend window (Friday, Saturday, Sunday)
    df['is_weekend'] = df['dayofweek'].apply(lambda dw: 1 if dw in [4, 5, 6] else 0)
    return df

def train_and_save_prophet():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found.")
        return

    # 1. Load and Parse Dataset
    df = pd.read_csv(CSV_FILE)
    df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
    df = df.dropna().sort_values('ds').reset_index(drop=True)
    
    print(f"📊 Dataset Loaded: {len(df)} total entries from {df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}")

    # 2. Filter Extreme Operational Outliers (< ₱1,800)
    # Excludes extreme flood/disruption days to prevent baseline distortion
    df_clean = df[df['y'] >= 1800].copy()
    
    # 3. Apply Log-Transformation to $y$ to Stabilize Variance
    df_clean['y_orig'] = df_clean['y']
    df_clean['y'] = np.log(df_clean['y'])

    # 4. Add Regressors
    df_clean = add_custom_features(df_clean)

    # 5. Configure & Fit Meta Prophet Model
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=0.5
    )
    
    # Add custom extra regressors
    model.add_regressor('is_payday')
    model.add_regressor('is_weekend')
    
    model.fit(df_clean)

    # 6. Evaluate In-Sample Predictions (Reverse Log Transform)
    in_sample = model.predict(df_clean)
    in_sample['yhat_actual'] = np.exp(in_sample['yhat'])
    
    mae = mean_absolute_error(df_clean['y_orig'], in_sample['yhat_actual'])
    rmse = np.sqrt(mean_squared_error(df_clean['y_orig'], in_sample['yhat_actual']))
    mape = np.mean(np.abs((df_clean['y_orig'] - in_sample['yhat_actual']) / df_clean['y_orig'])) * 100

    print("\n=== 📈 Meta Prophet Training Evaluation ===")
    print(f"Mean Absolute Error (MAE)     : ₱{mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ₱{rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # 7. Generate 90-Day Future Forecast Dataframe
    future = model.make_future_dataframe(periods=90, freq='D')
    future = add_custom_features(future)
    
    forecast = model.predict(future)
    forecast['yhat'] = np.exp(forecast['yhat']) # Convert back from log scale
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

    # 8. Save Model and Forecast Cache
    joblib.dump(model, MODEL_OUTPUT)
    joblib.dump(forecast, FORECAST_OUTPUT)
    print(f"\n✅ Saved Optimized Meta Prophet Model -> `{MODEL_OUTPUT}`")
    print(f"✅ Saved 90-Day Forecast Cache -> `{FORECAST_OUTPUT}`")

if __name__ == '__main__':
    train_and_save_prophet()
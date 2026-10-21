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
    """
    Adds Payday and Weekend binary indicators to capture retail store demand surges.
    """
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    
    # Philippine Payday window (14th-16th and 28th-31st)
    df['is_payday'] = df['day'].apply(lambda d: 1 if (14 <= d <= 16) or (28 <= d <= 31) else 0)
    # Weekend window (Friday, Saturday, Sunday)
    df['is_weekend'] = df['dayofweek'].apply(lambda dw: 1 if dw in [4, 5, 6] else 0)
    
    return df

def train_and_save_prophet():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found.")
        return

    # 1. Load and Parse Dataset
    df = pd.read_csv(CSV_FILE)
    
    # Map column names
    if 'Date' in df.columns:
        df['Date'] = df['Date'].ffill()
        df['ds'] = pd.to_datetime(df['Date'])
    elif 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
    else:
        print("❌ Error: No valid date column found ('Date' or 'ds').")
        return

    if 'Daily_Total_Sales' in df.columns:
        df['y'] = pd.to_numeric(df['Daily_Total_Sales'], errors='coerce')
    elif 'y' in df.columns:
        df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Deduplicate daily records to aggregate total revenue per date
    df_daily = df.groupby('ds', as_index=False)['y'].first()
    df_daily = df_daily.dropna().sort_values('ds').reset_index(drop=True)

    if df_daily.empty:
        print("❌ Error: No valid numeric sales data found.")
        return

    # ----------------------------------------------------
    # ACCURACY IMPROVEMENT: Fill date gaps with linear interpolation
    # ----------------------------------------------------
    full_date_range = pd.date_range(start=df_daily['ds'].min(), end=df_daily['ds'].max(), freq='D')
    df_daily = df_daily.set_index('ds').reindex(full_date_range).rename_axis('ds').reset_index()
    df_daily['y'] = df_daily['y'].interpolate(method='linear').fillna(0)
    
    print(f"📊 Dataset Processed: {len(df_daily)} continuous daily entries from {df_daily['ds'].min().strftime('%Y-%m-%d')} to {df_daily['ds'].max().strftime('%Y-%m-%d')}")

    # 2. Add Custom Features
    df_clean = add_custom_features(df_daily.copy())

    # 3. Configure & Fit High-Accuracy Prophet Model
    model = Prophet(
        growth='linear',
        yearly_seasonality=False,      # Set True if dataset spans >1-2 years
        weekly_seasonality=True,       # High day-of-week sensitivity
        daily_seasonality=False,
        seasonality_mode='multiplicative', # Scales weekly/payday variations proportionally to revenue
        changepoint_prior_scale=0.08,  # Tuned scale (0.08) prevents overfitting on temporary noise
        seasonality_prior_scale=10.0,  # Strongly fits day-of-week patterns
        holidays_prior_scale=10.0,
        interval_width=0.95
    )

    # ACCURACY IMPROVEMENT: Add Philippine Country Holidays
    try:
        model.add_country_holidays(country_name='PH')
        print("🇵🇭 Applied Philippine National Holidays to Prophet Model.")
    except Exception as e:
        print(f"⚠️ Notice: PH Holidays omitted ({e})")

    # Add regressors
    model.add_regressor('is_payday', prior_scale=10.0)
    model.add_regressor('is_weekend', prior_scale=10.0)
    
    model.fit(df_clean)

    # 4. In-Sample Evaluation
    in_sample = model.predict(df_clean)
    
    # Clip negative values
    in_sample['yhat'] = in_sample['yhat'].clip(lower=0)
    
    mae = mean_absolute_error(df_clean['y'], in_sample['yhat'])
    rmse = np.sqrt(mean_squared_error(df_clean['y'], in_sample['yhat']))
    
    # Safe MAPE calculation ignoring zeros
    non_zero_mask = df_clean['y'] > 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((df_clean.loc[non_zero_mask, 'y'] - in_sample.loc[non_zero_mask, 'yhat']) / df_clean.loc[non_zero_mask, 'y'])) * 100
    else:
        mape = 0.0

    print("\n=== 📈 Meta Prophet Retrained Evaluation ===")
    print(f"Mean Absolute Error (MAE)     : ₱{mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ₱{rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # 5. Generate 90-Day Forecast
    future = model.make_future_dataframe(periods=90, freq='D')
    future = add_custom_features(future)
    
    forecast = model.predict(future)
    
    # Clip negative predictions
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    forecast['ds_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')

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

    # 6. Save Model Artifacts
    joblib.dump(model, MODEL_OUTPUT)
    joblib.dump(forecast, FORECAST_OUTPUT)
    print(f"\n✅ Retrained Meta Prophet Model saved to `{MODEL_OUTPUT}`")
    print(f"✅ Saved updated forecast cache to `{FORECAST_OUTPUT}`")

if __name__ == '__main__':
    train_and_save_prophet()
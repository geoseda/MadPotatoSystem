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
        print(f"❌ Error: {CSV_FILE} not found. Ensure the dataset is in the same directory.")
        return

    # 1. Load Data
    df = pd.read_csv(CSV_FILE)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    print(f"📊 Dataset Loaded: {len(df)} historical entries from {df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}")

    # 2. Configure & Fit Meta Prophet Model
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(df)

    # 3. In-Sample Performance Evaluation
    in_sample = model.predict(df[['ds']])
    mae = mean_absolute_error(df['y'], in_sample['yhat'])
    rmse = np.sqrt(mean_squared_error(df['y'], in_sample['yhat']))
    mape = np.mean(np.abs((df['y'] - in_sample['yhat']) / df['y'])) * 100

    print("\n=== 📈 Meta Prophet Training Evaluation ===")
    print(f"Mean Absolute Error (MAE)  : ₱{mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ₱{rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # 4. Generate 90-Day Future Forecast Dataframe
    future = model.make_future_dataframe(periods=90, freq='D')
    forecast = model.predict(future)
    forecast['ds_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')

    # 5. Save Model and Forecast Cache to Disk
    joblib.dump(model, MODEL_OUTPUT)
    joblib.dump(forecast, FORECAST_OUTPUT)
    print(f"\n✅ Saved Meta Prophet Model -> `{MODEL_OUTPUT}`")
    print(f"✅ Saved 90-Day Forecast Cache -> `{FORECAST_OUTPUT}`")

if __name__ == '__main__':
    train_and_save_prophet()
    
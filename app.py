from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from prophet import Prophet
import pandas as pd

app = Flask(__name__, template_folder='.')
CORS(app)

# In-memory database storing daily log records
logbook_db = {
    "2026-09-28": {"sales": 2100},
    "2026-09-29": {"sales": 2200},
    "2026-09-30": {"sales": 2050},
    "2026-10-01": {"sales": 3300},
    "2026-10-02": {"sales": 4200},
    "2026-10-03": {"sales": 5100},
    "2026-10-04": {"sales": 3900}
}

def evaluate_sales_performance(amount):
    """Categorizes daily revenue based on business owner parameters."""
    if amount < 2000:
        return "Operating Deficit 🚨 (Below ₱2,000 minimum operating cost)", "deficit"
    elif amount < 3000:
        return "Underperforming Day ⚠️ (Below ₱3,000 target)", "bad"
    elif amount < 3800:
        return "Below Average Day 📊 (Below acceptable range)", "below-avg"
    elif amount <= 4800:
        return "Acceptable / Typical Day ✅ (₱3,800 - ₱4,800)", "acceptable"
    else:
        return "High-Performing / Successful Day 🚀 (Above ₱5,000)", "successful"

def fit_prophet_model(periods=30):
    """Formats sales records and fits the Meta Prophet model."""
    records = []
    for date_str, entry in logbook_db.items():
        sales = entry.get("sales", 0)
        if sales > 0:
            records.append({"ds": date_str, "y": sales})

    if len(records) < 2:
        return None

    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify(logbook_db)

@app.route("/api/logs", methods=["POST"])
def save_log():
    data = request.json or {}
    date_str = data.get("date")
    sales = data.get("sales", 0)

    if not date_str:
        return jsonify({"status": "error", "message": "Date is required"}), 400

    logbook_db[date_str] = data
    return jsonify({"status": "success", "message": f"Log saved for {date_str}"})

@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    period = request.args.get("period", "next_week")
    periods_map = {"tomorrow": 1, "next_week": 7, "next_month": 30}
    days = periods_map.get(period, 7)

    forecast = fit_prophet_model(periods=days)

    if forecast is None:
        return jsonify({
            "status": "warning",
            "message": "Insufficient sales log data to generate Meta Prophet predictions.",
            "forecast": []
        })

    future_forecast = forecast.tail(days)
    results = []
    for _, row in future_forecast.iterrows():
        predicted_val = max(0, int(round(row["yhat"])))
        status_label, category = evaluate_sales_performance(predicted_val)
        results.append({
            "date": row["ds"].strftime("%Y-%m-%d"),
            "yhat": predicted_val,
            "yhat_lower": max(0, int(round(row["yhat_lower"]))),
            "yhat_upper": max(0, int(round(row["yhat_upper"]))),
            "performance": status_label,
            "category": category
        })

    return jsonify({"status": "success", "period": period, "forecast": results})

@app.route("/api/chat", methods=["POST"])
def chat_assistant():
    data = request.json or {}
    user_query = data.get("query", "").lower()

    forecast = fit_prophet_model(periods=30)

    if forecast is None:
        return jsonify({"reply": "🤖 **Meta Prophet Assistant:** Please record at least 2 daily sales entries in the logbook to generate predictions."})

    future_30 = forecast.tail(30)
    tomorrow_val = max(0, int(round(future_30.iloc[0]["yhat"])))
    tomorrow_status, _ = evaluate_sales_performance(tomorrow_val)

    next_week_val = max(0, int(round(future_30.head(7)["yhat"].sum())))
    weekly_avg = int(round(next_week_val / 7))
    weekly_status, _ = evaluate_sales_performance(weekly_avg)

    if "tomorrow" in user_query:
        reply = (
            f"🔮 **Meta Prophet Forecast for Tomorrow:** ₱{tomorrow_val:,}<br>"
            f"📌 **Status:** {tomorrow_status}"
        )
    elif "week" in user_query:
        reply = (
            f"📈 **Meta Prophet 7-Day Forecast:** ₱{next_week_val:,} (Avg ₱{weekly_avg:,}/day)<br>"
            f"📌 **Daily Performance Projection:** {weekly_status}"
        )
    elif "month" in user_query:
        next_month_val = max(0, int(round(future_30["yhat"].sum())))
        reply = f"🗓️ **Meta Prophet 30-Day Total Forecast:** ₱{next_month_val:,}"
    elif "targets" in user_query or "benchmark" in user_query or "good" in user_query or "bad" in user_query:
        reply = (
            "📌 **Mad Potato Sales Revenue Benchmarks:**<br>"
            "• **Deficit Min:** Below ₱2,000 (Loss/Operating Overhead)<br>"
            "• **Underperforming:** Below ₱3,000<br>"
            "• **Acceptable / Typical:** ₱3,800 – ₱4,800<br>"
            "• **High-Performing:** ₱5,000+"
        )
    else:
        reply = (
            f"🤖 **Meta Prophet Analytics Summary:**<br>"
            f"• Tomorrow's Forecast: ₱{tomorrow_val:,} ({tomorrow_status})<br>"
            f"• Next 7-Day Total: ₱{next_week_val:,} (Daily Avg: ₱{weekly_avg:,})"
        )

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
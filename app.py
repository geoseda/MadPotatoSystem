import os
import re
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from prophet import Prophet

app = Flask(__name__, template_folder='.')
CORS(app)

CSV_FILE = "sales_data.csv"
NLP_MODEL_FILE = "nlp_pipeline.joblib"

# ITEM CODE DICTIONARY (Mad Potato Operational Glossary)
ITEM_NAMES = {
    "M": "Medio (M)",
    "G": "Grande (G)",
    "J": "Jumbo (J)",
    "MG": "Mega (MG)",
    "2XL": "2XL",
    "3XL": "3XL",
    "JC": "Juice Cups (JC)",
    "F": "Fries (F)",
    "CP": "Chicken Poppers (CP)",
    "JJ": "Jumbo Juice (JJ)",
    "BW": "Bottled Water (BW)"
}

logbook_db = {}
nlp_model_pipeline = None

def load_nlp_model():
    global nlp_model_pipeline
    if os.path.exists(NLP_MODEL_FILE):
        nlp_model_pipeline = joblib.load(NLP_MODEL_FILE)
        print("✅ Loaded trained NLP model pipeline.")
    else:
        print("⚠️ NLP pipeline not found. Run train_nlp.py first.")

def load_initial_csv_data():
    global logbook_db
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            # Standardize date format from DD/MM/YYYY to YYYY-MM-DD
            df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
            
            for _, row in df.iterrows():
                date_str = row['ds'].strftime("%Y-%m-%d")
                sales_val = float(row['y'])
                logbook_db[date_str] = {"sales": sales_val, "top": {}, "bottom": {}}
            print(f"✅ Loaded {len(df)} historical entries from {CSV_FILE}")
        except Exception as e:
            print(f"Error loading {CSV_FILE}: {e}")

load_nlp_model()
load_initial_csv_data()

def evaluate_sales_performance(amount):
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
        daily_seasonality=False,
        seasonality_mode='multiplicative'
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
    try:
        csv_rows = []
        for d, entry in sorted(logbook_db.items()):
            s = entry.get("sales", 0)
            if s > 0:
                csv_rows.append({"ds": d, "y": s})
        pd.DataFrame(csv_rows).to_csv(CSV_FILE, index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")

    return jsonify({"status": "success", "message": f"Log saved for {date_str}"})

@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    period = request.args.get("period", "next_week")
    periods_map = {"tomorrow": 1, "next_week": 7, "next_month": 30}
    days = periods_map.get(period, 7)

    forecast = fit_prophet_model(periods=days)
    if forecast is None:
        return jsonify({"status": "warning", "forecast": []})

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
    user_query = data.get("query", "").strip()
    query_lower = user_query.lower()

    if not user_query:
        return jsonify({"reply": "Please enter a question regarding inventory, sales, or business benchmarks."})

    # Extract date if present in user query (e.g., "2026-10-04")
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', user_query)
    target_date = date_match.group(0) if date_match else None

    # Predict Intent using NLP Model
    predicted_intent = "general"
    if nlp_model_pipeline:
        predicted_intent = nlp_model_pipeline.predict([user_query])[0]

    # Force intent matching for explicit date/stock keywords
    if "stock" in query_lower or "inventory" in query_lower or "remaining" in query_lower:
        predicted_intent = "inventory_stocks"
    elif ("sales" in query_lower or "revenue" in query_lower) and target_date:
        predicted_intent = "date_sales"

    # ==========================================
    # 1. INVENTORY STOCKS INTENT
    # ==========================================
    if predicted_intent == "inventory_stocks":
        query_date = target_date or list(sorted(logbook_db.keys()))[-1] if logbook_db else None
        
        if not query_date or query_date not in logbook_db:
            return jsonify({"reply": f"📦 **Inventory Query:** No logged stock data found for date `{query_date}`."})

        log_entry = logbook_db[query_date]
        bottom_data = log_entry.get("bottom", {})
        top_data = log_entry.get("top", {})

        reply = f"📦 **Inventory & Stock Report for {query_date}:**<br>"
        found_any = False

        # Check closing stocks (bottom table)
        for code, full_name in ITEM_NAMES.items():
            if code in bottom_data:
                closing_qty = bottom_data[code].get("closing", 0)
                stocks_qty = bottom_data[code].get("stocks", 0)
                reply += f"• **{full_name}:** {closing_qty} closing stock (Stocked: {stocks_qty})<br>"
                found_any = True

        # Check item usage (top table)
        if top_data:
            reply += "<br>📋 **Item Usage Breakdown:**<br>"
            for code, full_name in ITEM_NAMES.items():
                if code in top_data:
                    usage = top_data[code].get("usage", 0)
                    reply += f"• **{full_name}:** {usage} units used<br>"

        if not found_any:
            reply += "No itemized closing stock records found. Please check the logbook page."

        return jsonify({"reply": reply})

    # ==========================================
    # 2. DATE-SPECIFIC SALES INTENT
    # ==========================================
    if predicted_intent == "date_sales":
        query_date = target_date or list(sorted(logbook_db.keys()))[-1] if logbook_db else None
        if query_date in logbook_db:
            sales_val = logbook_db[query_date].get("sales", 0)
            status_text, _ = evaluate_sales_performance(sales_val)
            reply = (
                f"💰 **Sales Report for {query_date}:**<br>"
                f"• **Total Revenue:** ₱{sales_val:,.2f}<br>"
                f"• **Performance:** {status_text}"
            )
        else:
            reply = f"⚠️ No historical sales log found for date `{query_date}`."
        return jsonify({"reply": reply})

    # ==========================================
    # 3. FORECAST & BUSINESS INTENTS
    # ==========================================
    forecast = fit_prophet_model(periods=30)
    if forecast is None:
        return jsonify({"reply": "🤖 **Business Assistant:** Please record sales in the logbook to enable analytics."})

    future_30 = forecast.tail(30)
    tomorrow_val = max(0, int(round(future_30.iloc[0]["yhat"])))
    tomorrow_status, _ = evaluate_sales_performance(tomorrow_val)
    next_week_val = max(0, int(round(future_30.head(7)["yhat"].sum())))
    weekly_avg = int(round(next_week_val / 7))
    weekly_status, _ = evaluate_sales_performance(weekly_avg)

    if predicted_intent == "tomorrow":
        reply = f"🔮 **Meta Prophet Forecast for Tomorrow:** ₱{tomorrow_val:,}<br>📌 **Status:** {tomorrow_status}"
    elif predicted_intent == "week":
        reply = f"📈 **7-Day Sales Outlook:** ₱{next_week_val:,} (Daily Avg ₱{weekly_avg:,})<br>📌 **Status:** {weekly_status}"
    elif predicted_intent == "month":
        next_month_val = max(0, int(round(future_30["yhat"].sum())))
        reply = f"🗓️ **30-Day Projected Revenue:** ₱{next_month_val:,}"
    elif predicted_intent == "targets":
        reply = (
            "🎯 **Mad Potato Business Performance Tiers:**<br>"
            "• **Operating Deficit:** < ₱2,000 (Loss / Low Traffic)<br>"
            "• **Underperforming:** ₱2,000 – ₱2,999<br>"
            "• **Acceptable Range:** ₱3,800 – ₱4,800<br>"
            "• **High-Performing:** ₱5,000+ (Peak Demand / Weekend)"
        )
    else:
        reply = (
            f"🤖 **Mad Potato Business & Inventory Assistant:**<br>"
            f"• Tomorrow's Forecast: ₱{tomorrow_val:,} ({tomorrow_status})<br>"
            f"• 7-Day Total Forecast: ₱{next_week_val:,}<br>"
            f"💡 *You can ask me about remaining inventory stocks (Medio, Jumbo, Chicken Poppers), daily sales, or business benchmarks!*"
        )

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
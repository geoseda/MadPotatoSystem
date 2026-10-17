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

ITEM_NAMES = {
    "M": "Medio (M)",
    "G": "Grande (G)",
    "J": "Jumbo (J)",
    "MG": "Mega (MG)",
    "ML": "Mega (ML)",
    "2XL": "2XL",
    "3XL": "3XL",
    "JC": "Juice Cups (JC)",
    "F": "Fries (F)",
    "CP": "Chicken Poppers (CP)",
    "JJ": "Jumbo Juice (JJ)",
    "BW": "Bottled Water (BW)"
}

ITEM_ALIASES = {
    "medio": "M",
    "grande": "G",
    "jumbo juice": "JJ",
    "jumbo": "J",
    "mega": "MG",
    "2xl": "2XL",
    "3xl": "3XL",
    "juice cups": "JC",
    "juice cup": "JC",
    "fries": "F",
    "fry": "F",
    "chicken poppers": "CP",
    "poppers": "CP",
    "bottled water": "BW",
    "water": "BW"
}

ALL_TOP_ITEMS = ['M', 'G', 'J', 'MG', '2XL', '3XL', 'JC', 'F', 'CP', 'JJ', 'BW']

logbook_db = {}
nlp_model_pipeline = None

def parse_val(val, default=0.0):
    if pd.isnull(val):
        return default
    val_str = str(val).strip()
    if not val_str:
        return default
    if '+' in val_str:
        try:
            return float(sum(float(x.strip()) for x in val_str.split('+') if x.strip()))
        except Exception:
            return default
    try:
        return float(val_str)
    except Exception:
        return default

def detect_specific_items(query_lower):
    matched_codes = []
    sorted_keywords = sorted(ITEM_ALIASES.keys(), key=len, reverse=True)
    temp_query = query_lower
    for kw in sorted_keywords:
        if kw in temp_query:
            code = ITEM_ALIASES[kw]
            if code not in matched_codes:
                matched_codes.append(code)
            temp_query = temp_query.replace(kw, "")
    return matched_codes

def load_nlp_model():
    global nlp_model_pipeline
    if os.path.exists(NLP_MODEL_FILE):
        try:
            nlp_model_pipeline = joblib.load(NLP_MODEL_FILE)
            print("Loaded trained NLP model pipeline.")
        except Exception as e:
            print(f"Error loading NLP model: {e}")
    else:
        print("NLP pipeline not found. Run train_nlp.py first.")

def load_initial_csv_data():
    global logbook_db
    logbook_db = {}
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            
            if 'Date' in df.columns or 'ds' in df.columns:
                date_col = 'Date' if 'Date' in df.columns else 'ds'
                df[date_col] = df[date_col].ffill()
                df['parsed_ds'] = pd.to_datetime(df[date_col])
                
                grouped = df.groupby('parsed_ds')
                for ds, group in grouped:
                    date_str = ds.strftime("%Y-%m-%d")
                    
                    sales_val = 0.0
                    if 'Daily_Total_Sales' in group.columns and pd.notnull(group['Daily_Total_Sales'].iloc[0]):
                        sales_val = parse_val(group['Daily_Total_Sales'].iloc[0], 0.0)
                    elif 'y' in group.columns and pd.notnull(group['y'].iloc[0]):
                        sales_val = parse_val(group['y'].iloc[0], 0.0)
                    
                    top_data = {}
                    bottom_data = {}
                    
                    for _, row in group.iterrows():
                        item_code = str(row.get('Item_Code', '')).strip()
                        if item_code == 'ML':
                            item_code = 'MG'
                        
                        if item_code:
                            top_data[item_code] = {
                                "starting": parse_val(row.get('Starting_Stock'), 0.0),
                                "end": parse_val(row.get('Ending_Stock'), 0.0),
                                "usage": parse_val(row.get('Daily_Usage'), 0.0)
                            }
                            
                            bottom_data[item_code] = {
                                "opening": str(row.get('Opening', '')).strip() if pd.notnull(row.get('Opening')) else "",
                                "stocks": parse_val(row.get('Stocks'), parse_val(row.get('Restock_Added'), 0.0)),
                                "closing": parse_val(row.get('Closing'), parse_val(row.get('Closing_Stock'), 0.0))
                            }
                    
                    logbook_db[date_str] = {
                        "date": date_str,
                        "sales": sales_val,
                        "top": top_data,
                        "bottom": bottom_data
                    }
                    
            print(f"Loaded {len(logbook_db)} historical days from '{CSV_FILE}'")
        except Exception as e:
            print(f"Error loading dataset {CSV_FILE}: {e}")

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
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.35,
        seasonality_prior_scale=2.0
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast

def save_logbook_to_csv():
    rows = []
    for date_str in sorted(logbook_db.keys()):
        entry = logbook_db[date_str]
        sales = entry.get("sales", 0)
        top = entry.get("top", {})
        bottom = entry.get("bottom", {})

        first_row = True
        for item in ALL_TOP_ITEMS:
            t_item = top.get(item, {})
            b_item = bottom.get(item, {})

            starting = t_item.get("starting", 0.0)
            ending = t_item.get("end", 0.0)
            usage = t_item.get("usage", 0.0)

            opening = b_item.get("opening", "")
            stocks = b_item.get("stocks", 0.0)
            closing = b_item.get("closing", 0.0)

            rows.append({
                "Date": date_str if first_row else "",
                "Item_Code": item,
                "Starting_Stock": starting,
                "Ending_Stock": ending,
                "Daily_Usage": usage,
                "Restock_Added": stocks,
                "Closing_Stock": closing,
                "Daily_Total_Sales": sales,
                "Opening": opening,
                "Stocks": stocks,
                "Closing": closing
            })
            first_row = False

    df_out = pd.DataFrame(rows)
    df_out.to_csv(CSV_FILE, index=False)
    print(f"Saved logbook data to '{CSV_FILE}'")

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

    if not date_str:
        return jsonify({"status": "error", "message": "Date is required"}), 400

    logbook_db[date_str] = data
    try:
        save_logbook_to_csv()
    except Exception as e:
        print(f"Error saving to {CSV_FILE}: {e}")

    return jsonify({"status": "success", "message": f"Log saved to {CSV_FILE} for {date_str}"})

@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    period = request.args.get("period", "next_week")
    periods_map = {"tomorrow": 1, "next_week": 7, "next_month": 30}
    days = periods_map.get(period, 7)

    forecast = fit_prophet_model(periods=days)
    if forecast is None:
        return jsonify({"status": "warning", "forecast": []})

    all_dates = sorted(logbook_db.keys())
    results = []
    
    recent_historical_dates = all_dates[-5:] if len(all_dates) >= 5 else all_dates
    for date_str in recent_historical_dates:
        sales_val = logbook_db[date_str].get("sales", 0)
        status_label, category = evaluate_sales_performance(sales_val)
        results.append({
            "date": date_str,
            "yhat": int(round(sales_val)),
            "yhat_lower": int(round(sales_val)),
            "yhat_upper": int(round(sales_val)),
            "performance": status_label,
            "category": category,
            "is_actual": True
        })

    future_forecast = forecast.tail(days)
    for _, row in future_forecast.iterrows():
        f_date = row["ds"].strftime("%Y-%m-%d")
        if f_date not in [r["date"] for r in results]:
            predicted_val = max(0, int(round(row["yhat"])))
            status_label, category = evaluate_sales_performance(predicted_val)
            results.append({
                "date": f_date,
                "yhat": predicted_val,
                "yhat_lower": max(0, int(round(row["yhat_lower"]))),
                "yhat_upper": max(0, int(round(row["yhat_upper"]))),
                "performance": status_label,
                "category": category,
                "is_actual": False
            })

    return jsonify({"status": "success", "period": period, "forecast": results})

@app.route("/api/chat", methods=["POST"])
def chat_assistant():
    data = request.json or {}
    user_query = data.get("query", "").strip()
    query_lower = user_query.lower()

    if not user_query:
        return jsonify({"reply": "Please enter a question regarding inventory, sales, or business benchmarks."})

    date_match = re.search(r'\d{4}-\d{2}-\d{2}', user_query)
    target_date = date_match.group(0) if date_match else None

    target_items = detect_specific_items(query_lower)

    recognized_keywords = [
        "sales", "revenue", "earn", "earned", "make", "made", "income",
        "stock", "stocks", "inventory", "remaining", "restock", "low", "need", "order",
        "today", "tomorrow", "week", "month", "forecast", "predict", "projection",
        "target", "targets", "benchmark", "benchmarks", "tier", "tiers", "performance"
    ]
    has_recognized_keyword = any(k in query_lower for k in recognized_keywords)

    if not target_date and not has_recognized_keyword and not target_items:
        return jsonify({
            "reply": f"Sorry, I couldn't recognize your query '{user_query}'. Please ask a question about daily sales (e.g., 'sales on 2026-10-16'), inventory stocks (e.g., 'remaining stock for Medio'), AI forecasts, or business benchmarks."
        })

    predicted_intent = "general"
    if nlp_model_pipeline:
        try:
            predicted_intent = nlp_model_pipeline.predict([user_query])[0]
        except Exception:
            pass

    if target_items or any(k in query_lower for k in ["stock", "inventory", "remaining", "restock", "low", "need"]):
        predicted_intent = "inventory_stocks"
    elif ("sales" in query_lower or "revenue" in query_lower or "earn" in query_lower or "made" in query_lower) and target_date:
        predicted_intent = "date_sales"

    # 1. DATE-SPECIFIC SALES INTENT
    if predicted_intent == "date_sales" or (target_date and any(k in query_lower for k in ["sales", "revenue", "earn", "made"])):
        query_date = target_date
        if query_date in logbook_db:
            sales_val = logbook_db[query_date].get("sales", 0)
            status_text, _ = evaluate_sales_performance(sales_val)
            reply = (
                f"💰 <b>Sales Report for {query_date}:</b><br>"
                f"• Total Revenue: ₱{sales_val:,.2f}<br>"
                f"• Performance: {status_text}"
            )
        else:
            reply = f"Sorry, there is no sales information recorded for the date {query_date} in our dataset."
        return jsonify({"reply": reply})

    # 2. ITEM-SPECIFIC OR OVERALL INVENTORY INTENT
    if predicted_intent == "inventory_stocks":
        query_date = target_date or (list(sorted(logbook_db.keys()))[-1] if logbook_db else None)
        
        if not query_date or query_date not in logbook_db:
            return jsonify({"reply": f"Sorry, there is no inventory stock record found for the date {query_date}."})

        log_entry = logbook_db[query_date]
        top_data = log_entry.get("top", {})
        bottom_data = log_entry.get("bottom", {})

        # Item-specific query pulled from END column (Ending_Stock)
        if target_items:
            reply = f"📦 Inventory Report for {query_date}:<br><br>"
            for code in target_items:
                full_name = ITEM_NAMES.get(code, code)
                end_qty = int(round(top_data.get(code, {}).get("end", 0)))
                reply += f"• {full_name}:<br>Stocks: {end_qty} cups<br><br>"

            return jsonify({"reply": reply})

        # Restock alert
        is_restock_query = any(k in query_lower for k in ["restock", "low", "need", "order"])
        if is_restock_query:
            reply = f"🚨 Restock Alert for {query_date}:<br><br>"
            low_items = []

            for code, full_name in ITEM_NAMES.items():
                end_qty = int(round(top_data.get(code, {}).get("end", 0)))
                if end_qty <= 10:
                    low_items.append((full_name, end_qty))

            if low_items:
                reply += "Items Needing Restock (10 cups or less):<br>"
                for name, qty in low_items:
                    reply += f"• {name}: {qty} cups<br>"
            else:
                reply += "All item stocks are currently healthy.<br>"

            return jsonify({"reply": reply})

        # Overall Inventory Summary
        reply = f"📦 Inventory Report for {query_date}:<br>"
        for code, full_name in ITEM_NAMES.items():
            if code in top_data:
                end_qty = int(round(top_data[code].get("end", 0)))
                reply += f"• {full_name}: {end_qty} cups<br>"

        return jsonify({"reply": reply})

    # 3. FORECAST & BUSINESS INTENTS
    forecast = fit_prophet_model(periods=30)
    if forecast is None:
        return jsonify({"reply": "Business Assistant: Please record sales in the logbook to enable analytics."})

    future_30 = forecast.tail(30)
    tomorrow_val = max(0, int(round(future_30.iloc[0]["yhat"])))
    tomorrow_status, _ = evaluate_sales_performance(tomorrow_val)
    next_week_val = max(0, int(round(future_30.head(7)["yhat"].sum())))
    weekly_avg = int(round(next_week_val / 7))
    weekly_status, _ = evaluate_sales_performance(weekly_avg)

    if predicted_intent == "tomorrow":
        reply = f"🔮 Meta Prophet Forecast for Tomorrow: ₱{tomorrow_val:,}<br>Status: {tomorrow_status}"
    elif predicted_intent == "week":
        reply = f"📈 7-Day Sales Outlook: ₱{next_week_val:,} (Daily Avg ₱{weekly_avg:,})<br>Status: {weekly_status}"
    elif predicted_intent == "month":
        next_month_val = max(0, int(round(future_30["yhat"].sum())))
        reply = f"🗓️ 30-Day Projected Revenue: ₱{next_month_val:,}"
    elif predicted_intent == "targets":
        reply = (
            "🎯 Mad Potato Performance Tiers:<br>"
            "• Operating Deficit: < ₱2,000<br>"
            "• Underperforming: ₱2,000 – ₱2,999<br>"
            "• Acceptable Range: ₱3,800 – ₱4,800<br>"
            "• High-Performing: ₱5,000+"
        )
    else:
        reply = (
            f"🤖 Mad Potato Business & Inventory Assistant:<br>"
            f"• Tomorrow's Forecast: ₱{tomorrow_val:,}<br>"
            f"• 7-Day Total Forecast: ₱{next_week_val:,}<br>"
            f"You can ask me about stock for specific items (e.g. 'stock for Medio'), daily sales, or forecasts!"
        )

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
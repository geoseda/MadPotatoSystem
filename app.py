import os
import re
import threading
from datetime import datetime, timedelta
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

# Items that should NOT have "cups" appended as units
NON_CUP_ITEMS = {'F', 'CP', 'JJ', 'BW'}

def get_unit(code):
    return "" if code in NON_CUP_ITEMS else " cups"

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

# ----------------------------------------------------
# FLEXIBLE DATE UNDERSTANDING
# ----------------------------------------------------
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12
}
_MONTH_RX = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"

def _safe_date(year, month, day):
    try:
        return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
    except ValueError:
        return None

def _pick_year_for(month, day):
    this_year = datetime.now().year
    years = [this_year]
    if logbook_db:
        last_year = int(max(logbook_db.keys())[:4])
        if last_year not in years:
            years.append(last_year)
    for y in years:
        d = _safe_date(y, month, day)
        if d and d in logbook_db:
            return y
    return this_year

def extract_date(query_lower):
    # 1) ISO: 2026-10-16
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", query_lower)
    if m:
        d = _safe_date(m.group(1), m.group(2), m.group(3))
        if d:
            return d, m.group(0)

    # 2) Slashes: 10/16/2026
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", query_lower)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        month, day = (b, a) if a > 12 else (a, b)
        d = _safe_date(y, month, day)
        if d:
            return d, m.group(0)

    # 3) "October 16", "Oct 16th, 2026"
    m = re.search(_MONTH_RX + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", query_lower)
    if m:
        month, day = MONTHS[m.group(1)], int(m.group(2))
        year = int(m.group(3)) if m.group(3) else _pick_year_for(month, day)
        d = _safe_date(year, month, day)
        if d:
            return d, m.group(0)

    # 4) "16 October", "16th of Oct 2026"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_RX + r"\b\.?(?:,?\s+(\d{4}))?", query_lower)
    if m:
        day, month = int(m.group(1)), MONTHS[m.group(2)]
        year = int(m.group(3)) if m.group(3) else _pick_year_for(month, day)
        d = _safe_date(year, month, day)
        if d:
            return d, m.group(0)

    # 5) Relative: yesterday / today
    m = re.search(r"\byesterday\b", query_lower)
    if m:
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), m.group(0)
    
    m = re.search(r"\btoday\b", query_lower)
    if m:
        return datetime.now().strftime("%Y-%m-%d"), m.group(0)

    return None, None

def pretty_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %d, %Y")

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
        return jsonify({"reply": "Please enter a question regarding inventory, sales, or business benchmarks.", "is_stock_updated": False})

    target_date, date_span = extract_date(query_lower)
    query_no_date = query_lower.replace(date_span, " ") if date_span else query_lower
    target_items = detect_specific_items(query_no_date)

    # ----------------------------------------------------
    # 1. CLOSING / DIRECT SET VALUE COMMAND MATCHING
    # e.g., "closing/update cups today for medio is 5", "closing medio is 5", "set medio stock to 5"
    # ----------------------------------------------------
    closing_regex = re.compile(
        r'\b(closing|close|set|update|is|to)\b'
        r'(?:[a-z\s]+)?\b(\d+)\b'
    )
    is_closing_command = any(k in query_no_date for k in ["closing", "close", "set"]) or ("update" in query_no_date and "is" in query_no_date)
    closing_match = closing_regex.search(query_no_date)

    if is_closing_command and target_items:
        # Extract number
        digits = re.findall(r'\b\d+\b', query_no_date)
        if digits:
            exact_qty = float(digits[-1])
            query_date = target_date or datetime.now().strftime("%Y-%m-%d")
            
            # Ensure date entry exists in logbook_db
            if query_date not in logbook_db:
                logbook_db[query_date] = {"date": query_date, "sales": 0.0, "top": {}, "bottom": {}}

            log_entry = logbook_db[query_date]
            top_data = log_entry.get("top", {})
            bottom_data = log_entry.get("bottom", {})

            reply_lines = []

            for code in target_items:
                full_name = ITEM_NAMES.get(code, code)
                unit_str = get_unit(code)

                t_item = top_data.get(code, {"starting": 0.0, "end": 0.0, "usage": 0.0})
                b_item = bottom_data.get(code, {"opening": "", "stocks": 0.0, "closing": 0.0})

                start_qty = float(t_item.get("starting", 0.0))
                
                # Set closing and end values directly
                t_item["end"] = exact_qty
                b_item["closing"] = exact_qty
                
                # Calculate daily usage
                usage_qty = max(0.0, start_qty - exact_qty)
                t_item["usage"] = usage_qty

                reply_lines.append(
                    f"📝 <b>Closing Logged!</b> {full_name}<br>"
                    f"• Date: <b>{query_date}</b><br>"
                    f"• End / Closing Stock: <b>{int(round(exact_qty))}</b>{unit_str}<br>"
                    f"• Calculated Usage: <b>{int(round(usage_qty))}</b>{unit_str}"
                )

                top_data[code] = t_item
                bottom_data[code] = b_item

            logbook_db[query_date]["top"] = top_data
            logbook_db[query_date]["bottom"] = bottom_data

            def _delayed_save():
                try:
                    save_logbook_to_csv()
                except Exception as e:
                    print(f"CSV sync error: {e}")
            threading.Timer(1.5, _delayed_save).start()

            return jsonify({
                "reply": "<br><br>".join(reply_lines),
                "is_stock_updated": True
            })

    # ----------------------------------------------------
    # 2. ADD / REMOVE RELATIVE QUANTITY MATCHING
    # ----------------------------------------------------
    action_regex = re.compile(
        r'\b(add|added|restock|plus|stock in|remove|removed|deduct|subtract|reduce|minus|take away|use)\b'
        r'\s*(?:[a-z]+\s+){0,2}?(\d+)\b'
    )
    action_match = action_regex.search(query_no_date)

    is_add_action = False
    is_remove_action = False
    quantity = None
    if action_match:
        action_word = action_match.group(1)
        quantity = int(action_match.group(2))
        if action_word in ("add", "added", "restock", "plus", "stock in"):
            is_add_action = True
        else:
            is_remove_action = True

    if (is_add_action or is_remove_action) and quantity is not None and not target_items:
        return jsonify({
            "reply": "🤔 I understood you want to " + ("add" if is_add_action else "remove") +
                     f" {quantity} units, but I couldn't tell which item. "
                     "Try: 'add 50 cups of medio', 'remove 20 grande', or 'closing cups today for medio is 5'.",
            "is_stock_updated": False
        })

    if (is_add_action or is_remove_action) and target_items and quantity is not None:
        query_date = target_date or (list(sorted(logbook_db.keys()))[-1] if logbook_db else datetime.now().strftime("%Y-%m-%d"))
        
        if query_date not in logbook_db:
            logbook_db[query_date] = {"date": query_date, "sales": 0.0, "top": {}, "bottom": {}}

        log_entry = logbook_db[query_date]
        top_data = log_entry.get("top", {})
        bottom_data = log_entry.get("bottom", {})

        reply_lines = []

        for code in target_items:
            full_name = ITEM_NAMES.get(code, code)
            unit_str = get_unit(code)
            
            t_item = top_data.get(code, {"starting": 0.0, "end": 0.0, "usage": 0.0})
            b_item = bottom_data.get(code, {"opening": "", "stocks": 0.0, "closing": 0.0})

            current_end = float(t_item.get("end", 0.0))
            current_stocks = float(b_item.get("stocks", 0.0))

            if is_add_action:
                new_end = current_end + quantity
                new_stocks = current_stocks + quantity
                reply_lines.append(
                    f"✅ <b>Stock added!</b> {full_name}<br>"
                    f"• Added: <b>+{quantity}</b>{unit_str}<br>"
                    f"• Remaining stock on {query_date}: {int(round(current_end))} → <b>{int(round(new_end))}</b>{unit_str}"
                )
            else:
                new_end = max(0.0, current_end - quantity)
                new_stocks = max(0.0, current_stocks - quantity)
                note = ""
                if quantity > current_end:
                    note = f"<br>⚠️ Only {int(round(current_end))} were in stock, so it was set to 0."
                reply_lines.append(
                    f"✅ <b>Stock removed!</b> {full_name}<br>"
                    f"• Removed: <b>-{quantity}</b>{unit_str}<br>"
                    f"• Remaining stock on {query_date}: {int(round(current_end))} → <b>{int(round(new_end))}</b>{unit_str}"
                    f"{note}"
                )

            t_item["end"] = new_end
            b_item["stocks"] = new_stocks

            top_data[code] = t_item
            bottom_data[code] = b_item

        logbook_db[query_date]["top"] = top_data
        logbook_db[query_date]["bottom"] = bottom_data

        def _delayed_save():
            try:
                save_logbook_to_csv()
            except Exception as e:
                print(f"CSV sync error: {e}")
        threading.Timer(1.5, _delayed_save).start()

        return jsonify({
            "reply": "<br><br>".join(reply_lines),
            "is_stock_updated": True
        })

    # ----------------------------------------------------
    # INTENT & GENERAL CHAT
    # ----------------------------------------------------
    recognized_keywords = [
        "sales", "revenue", "earn", "earned", "make", "made", "income",
        "stock", "stocks", "inventory", "remaining", "restock", "low", "need", "order",
        "today", "tomorrow", "week", "month", "forecast", "predict", "projection",
        "target", "targets", "benchmark", "benchmarks", "tier", "tiers", "performance"
    ]
    has_recognized_keyword = any(k in query_lower for k in recognized_keywords)

    if not target_date and not has_recognized_keyword and not target_items:
        return jsonify({
            "reply": f"Sorry, I couldn't recognize your query '{user_query}'. Please try prompts like 'closing cups today for medio is 5', 'add 50 cups of medio', or ask 'what is the sales for October 16?'.",
            "is_stock_updated": False
        })

    predicted_intent = "general"
    if nlp_model_pipeline:
        try:
            predicted_intent = nlp_model_pipeline.predict([user_query])[0]
        except Exception:
            pass

    sales_words = ["sales", "sale", "revenue", "earn", "made", "make", "income"]
    asks_sales = any(re.search(r"\b" + w, query_lower) for w in sales_words)

    if asks_sales and target_date and not target_items:
        predicted_intent = "date_sales"
    elif target_items or any(k in query_lower for k in ["stock", "inventory", "remaining", "restock", "low", "need"]):
        predicted_intent = "inventory_stocks"

    # 1. DATE-SPECIFIC SALES INTENT
    if predicted_intent == "date_sales":
        query_date = target_date
        note = ""
        if not query_date:
            if logbook_db:
                query_date = sorted(logbook_db.keys())[-1]
                note = "<br><i>(No date given, so I used your latest logged day.)</i>"
            else:
                return jsonify({"reply": "Which date? Try: 'sales for October 16' or 'sales on 2026-10-16'.", "is_stock_updated": False})

        if query_date in logbook_db and logbook_db[query_date].get("sales", 0) > 0:
            sales_val = logbook_db[query_date].get("sales", 0)
            status_text, _ = evaluate_sales_performance(sales_val)
            reply = (
                f"💰 <b>Sales Report for {pretty_date(query_date)}:</b><br>"
                f"• Total Revenue: <b>₱{sales_val:,.2f}</b><br>"
                f"• Performance: {status_text}{note}"
            )
        elif query_date in logbook_db:
            reply = (
                f"📒 {pretty_date(query_date)} is in the logbook, but no sales amount was recorded for it yet."
            )
        else:
            reply = f"Sorry, there is no sales record for {pretty_date(query_date)} in the logbook."
            try:
                last_date = sorted(logbook_db.keys())[-1] if logbook_db else None
                if last_date and query_date > last_date:
                    fc = fit_prophet_model(periods=90)
                    if fc is not None:
                        row = fc[fc["ds"] == pd.Timestamp(query_date)]
                        if not row.empty:
                            pred = max(0, int(round(row.iloc[0]["yhat"])))
                            status_text, _ = evaluate_sales_performance(pred)
                            reply = (
                                f"🔮 No recorded sales yet for {pretty_date(query_date)}, "
                                f"but the Meta Prophet forecast is <b>₱{pred:,}</b>.<br>"
                                f"• Status: {status_text}"
                            )
            except Exception as e:
                print(f"Forecast lookup error: {e}")
        return jsonify({"reply": reply, "is_stock_updated": False})

    # 2. ITEM-SPECIFIC OR OVERALL INVENTORY INTENT
    if predicted_intent == "inventory_stocks":
        query_date = target_date or (list(sorted(logbook_db.keys()))[-1] if logbook_db else None)
        
        if not query_date or query_date not in logbook_db:
            return jsonify({"reply": f"Sorry, there is no inventory stock record found for the date {query_date}.", "is_stock_updated": False})

        log_entry = logbook_db[query_date]
        top_data = log_entry.get("top", {})

        if target_items:
            reply = f"📦 Inventory Report for {query_date}:<br><br>"
            for code in target_items:
                full_name = ITEM_NAMES.get(code, code)
                unit_str = get_unit(code)
                end_qty = int(round(top_data.get(code, {}).get("end", 0)))
                reply += f"• {full_name}: {end_qty}{unit_str}<br>"

            return jsonify({"reply": reply, "is_stock_updated": False})

        is_restock_query = any(k in query_lower for k in ["restock", "low", "need", "order"])
        if is_restock_query:
            reply = f"🚨 Restock Alert for {query_date}:<br><br>"
            low_items = []

            for code, full_name in ITEM_NAMES.items():
                end_qty = int(round(top_data.get(code, {}).get("end", 0)))
                if end_qty <= 10:
                    low_items.append((code, full_name, end_qty))

            if low_items:
                reply += "Items Needing Restock (10 units or less):<br>"
                for code, name, qty in low_items:
                    unit_str = get_unit(code)
                    reply += f"• {name}: {qty}{unit_str}<br>"
            else:
                reply += "All item stocks are currently healthy.<br>"

            return jsonify({"reply": reply, "is_stock_updated": False})

        reply = f"📦 Inventory Report for {query_date}:<br>"
        for code, full_name in ITEM_NAMES.items():
            if code in top_data:
                unit_str = get_unit(code)
                end_qty = int(round(top_data[code].get("end", 0)))
                reply += f"• {full_name}: {end_qty}{unit_str}<br>"

        return jsonify({"reply": reply, "is_stock_updated": False})

    # 3. FORECAST & BUSINESS INTENTS
    forecast = fit_prophet_model(periods=30)
    if forecast is None:
        return jsonify({"reply": "Business Assistant: Please record sales in the logbook to enable analytics.", "is_stock_updated": False})

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
            f"🤖 MAPO ASSISTANT:<br>"
            f"• Tomorrow's Forecast: ₱{tomorrow_val:,}<br>"
            f"• 7-Day Total Forecast: ₱{next_week_val:,}<br>"
            f"You can ask me to modify stocks directly (e.g., 'closing cups today for medio is 5', 'add 50 cups of medio')!"
        )

    return jsonify({"reply": reply, "is_stock_updated": False})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
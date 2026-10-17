import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

MODEL_OUTPUT = 'nlp_pipeline.joblib'

# Expanded Dataset mapping natural language queries to Mad Potato business & inventory intents
TRAINING_DATA = [
    # 1. Intent: 'inventory_stocks' (Remaining levels, stock status, restock requirements)
    ("What are the remaining stocks for today?", "inventory_stocks"),
    ("How much stock do we have left?", "inventory_stocks"),
    ("What is the inventory level on 2026-10-16?", "inventory_stocks"),
    ("Check remaining stocks", "inventory_stocks"),
    ("What is the closing stock for Medio and Grande?", "inventory_stocks"),
    ("How many chicken poppers do we have left?", "inventory_stocks"),
    ("Show me remaining stock for 2XL and 3XL", "inventory_stocks"),
    ("What is our current inventory?", "inventory_stocks"),
    ("Do we have enough juice cups and bottled water?", "inventory_stocks"),
    ("Show remaining stocks for this date", "inventory_stocks"),
    ("How many fries are remaining?", "inventory_stocks"),
    ("Check jumbo juice inventory", "inventory_stocks"),
    ("What was the item usage on 2026-10-15?", "inventory_stocks"),
    ("How much stock did we restock?", "inventory_stocks"),
    ("Show stock breakdown", "inventory_stocks"),
    ("Which items are running low?", "inventory_stocks"),
    ("What items need restock?", "inventory_stocks"),
    ("What stocks need to be restocked?", "inventory_stocks"),
    ("Are we low on inventory?", "inventory_stocks"),
    ("Which products need to be restocked?", "inventory_stocks"),
    ("What do we need to order or restock?", "inventory_stocks"),
    ("Show me items that need restocking", "inventory_stocks"),
    ("What items are out of stock or low?", "inventory_stocks"),
    ("Which items are low in stock?", "inventory_stocks"),
    ("Restock list for today", "inventory_stocks"),

    # 2. Intent: 'date_sales' (Historical sales revenue for specific dates)
    ("What is the sales for this day?", "date_sales"),
    ("How much did we earn on 2026-10-16?", "date_sales"),
    ("What was the revenue on 2026-10-15?", "date_sales"),
    ("Show me total sales on this date", "date_sales"),
    ("How much revenue did we make yesterday?", "date_sales"),
    ("Sales for yesterday", "date_sales"),
    ("How much did we make on 2026-10-12?", "date_sales"),
    ("What was our sales on 2026-10-14?", "date_sales"),
    ("Show revenue for 2026-10-13", "date_sales"),
    ("What is the sales on 2026-05-10?", "date_sales"),
    ("Sales on 2026-08-20", "date_sales"),

    # 3. Intent: 'today' (Forecast for today)
    ("What is today's sales forecast?", "today"),
    ("Predict sales for today", "today"),
    ("How much revenue will we make today?", "today"),
    ("Today forecast", "today"),
    ("Current day prediction", "today"),
    ("Sales projection today", "today"),

    # 4. Intent: 'tomorrow' (Forecast for tomorrow)
    ("What is tomorrow's sales forecast?", "tomorrow"),
    ("Predict sales for tomorrow", "tomorrow"),
    ("How much revenue will we make tomorrow?", "tomorrow"),
    ("Tomorrow forecast", "tomorrow"),
    ("Next day prediction", "tomorrow"),

    # 5. Intent: 'week' (7-Day Forecast)
    ("What is the 7 day forecast?", "week"),
    ("Show me next week's sales", "week"),
    ("Weekly sales summary", "week"),
    ("7-day projection", "week"),
    ("How much will we earn next week?", "week"),

    # 6. Intent: 'month' (30-Day Forecast)
    ("What is the 30 day forecast?", "month"),
    ("Monthly prediction", "month"),
    ("Show me next month's total sales", "month"),
    ("30-day projection", "month"),

    # 7. Intent: 'targets' (Performance Tiers & Benchmarks)
    ("What are our operational targets?", "targets"),
    ("Show business benchmarks", "targets"),
    ("What is a high performing day?", "targets"),
    ("Deficit threshold", "targets"),
    ("How is the business performing?", "targets"),
    ("What are the target tiers?", "targets")
]

def train_and_save_nlp():
    X_train = [item[0] for item in TRAINING_DATA]
    y_train = [item[1] for item in TRAINING_DATA]

    # Pipeline: TF-IDF Vectorizer + Optimized Logistic Regression
    nlp_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(C=10.0, max_iter=500)
    )

    nlp_pipeline.fit(X_train, y_train)

    y_pred = nlp_pipeline.predict(X_train)
    print("=== 🤖 Mad Potato Business & Inventory Assistant Evaluation ===")
    print(f"Training Accuracy: {accuracy_score(y_train, y_pred) * 100:.2f}%\n")
    print(classification_report(y_train, y_pred, zero_division=0))

    joblib.dump(nlp_pipeline, MODEL_OUTPUT)
    print(f"✅ Saved NLP Business Assistant Model -> `{MODEL_OUTPUT}`")

if __name__ == '__main__':
    train_and_save_nlp()
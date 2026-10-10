import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

MODEL_OUTPUT = 'nlp_pipeline.joblib'

# Expanded Dataset mapping natural language queries to business & inventory intents
TRAINING_DATA = [
    # 1. Intent: 'inventory_stocks' (Querying stock levels, remaining items, closing stock)
    ("What are the remaining stocks for today?", "inventory_stocks"),
    ("How much stock do we have left?", "inventory_stocks"),
    ("What is the inventory level on 2026-10-10?", "inventory_stocks"),
    ("Check remaining stocks", "inventory_stocks"),
    ("What is the closing stock for Medio and Grande?", "inventory_stocks"),
    ("How many chicken poppers do we have left?", "inventory_stocks"),
    ("Show me remaining stock for 2XL and 3XL", "inventory_stocks"),
    ("What is our current inventory?", "inventory_stocks"),
    ("Do we have enough juice cups and bottled water?", "inventory_stocks"),
    ("Show remaining stocks for this date", "inventory_stocks"),

    # 2. Intent: 'date_sales' (Querying historical sales revenue on a specific date)
    ("What is the sales for this day?", "date_sales"),
    ("How much did we earn on 2026-10-09?", "date_sales"),
    ("What was the revenue on 2026-10-08?", "date_sales"),
    ("Show me total sales on this date", "date_sales"),
    ("How much revenue did we make yesterday?", "date_sales"),
    ("Sales for yesterday", "date_sales"),

    # 3. Intent: 'today' (Forecast for October 10)
    ("What is today's sales forecast?", "today"),
    ("Predict sales for today", "today"),
    ("How much revenue will we make today?", "today"),
    ("Today forecast", "today"),
    ("Current day prediction", "today"),

    # 4. Intent: 'tomorrow' (Forecast for October 11)
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

    # 6. Intent: 'month' (30-Day Forecast)
    ("What is the 30 day forecast?", "month"),
    ("Monthly prediction", "month"),
    ("Show me next month's total sales", "month"),

    # 7. Intent: 'targets' (Performance Tiers & Business Benchmarks)
    ("What are our operational targets?", "targets"),
    ("Show business benchmarks", "targets"),
    ("What is a high performing day?", "targets"),
    ("Deficit threshold", "targets"),
    ("How is the business performing?", "targets")
]

def train_and_save_nlp():
    X_train = [item[0] for item in TRAINING_DATA]
    y_train = [item[1] for item in TRAINING_DATA]

    # Pipeline: TF-IDF Vectorizer + Logistic Regression
    nlp_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words='english'),
        LogisticRegression(solver='lbfgs')
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
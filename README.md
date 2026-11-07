# 🍟 Mad Potato Management System

**Digitized Daily Operations and Predictive Sales Management System for Mad Potato (Teresa Branch)**

A web application that replaces paper logbooks with a digital daily logbook, backs up every entry to Google Sheets, and forecasts daily, weekly, and monthly sales using Meta Prophet. It also includes **MAPO**, a chat assistant that answers stock and forecast questions.

Built as a final project in **CPE 412: Human-AI Interaction**, Bachelor of Science in Computer Engineering, Technological Institute of the Philippines, Quezon City.

🔗 **Live demo:** https://madpotatosystem-2.onrender.com/
*(Hosted on a free tier, so the first load may take a minute to wake up.)*

---

---

## 🎯 The Problem

The Mad Potato Teresa branch recorded daily starting stock, item usage, closing inventory, and sales on paper logbooks and tally sheets. This was:

- Time-consuming and prone to calculation errors
- Vulnerable to physical record loss, with no off-site backup
- Unable to give management remote visibility or any forward-looking sales projection

## ✨ Features

- **Digitized Logbook:** records starting stock, ending stock, restocks, and total sales for 11 SKUs, and automatically computes item usage.
- **Admin Dashboard:** shows sales logged today, tomorrow's forecast, a 7-day average, cup stock levels, and recent sales.
- **Sales Forecasting:** next day, next 7 days, and next 30 days projections with a likely range for each day.
- **Performance Tiers:** every day is classified as Operating Deficit, Underperforming, Below Average, Acceptable, or High-Performing.
- **MAPO Assistant:** a natural-language chat for questions like "What is tomorrow's forecast?", "sales for October 16", or "closing cups today for medio is 5".
- **Google Sheets Sync:** each saved entry is sent to a Google Sheet through an Apps Script web app webhook for off-site backup.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| Forecasting | Meta Prophet (multiplicative seasonality), pandas |
| NLP | scikit-learn: TF-IDF + Multinomial Logistic Regression, regex entity rules |
| Frontend | HTML, CSS, JavaScript |
| Cloud backup | Google Sheets + Google Apps Script |
| Deployment | Render |

## 🏗️ System Architecture

```
Browser client (Dashboard, Forecast, Logbook, MAPO, Settings)
        │  HTTP                                  │ webhook POST
        ▼                                        ▼
Flask backend (app.py)                    Google Sheets (Apps Script)
 ├── /api/chat      → NLP pipeline (TF-IDF + LR + regex rules)
 ├── /api/logs      → Data store (in-memory logbook, sales_data.csv)
 └── /api/forecast  → Prophet engine (refit per request, tier labels)
```
## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/geoseda/MadPotatoSystem.git
cd MadPotatoSystem

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

### Retrain or re-evaluate the models

```bash
python train_nlp.py        # trains and saves nlp_pipeline.joblib
python train_prophet.py    # trains and saves the Prophet model and forecast
python evaluate_nlp.py     # intent classifier metrics
python evaluate_forecast.py  # Prophet vs baseline backtest
```

### Google Sheets sync (optional)

1. Deploy a Google Apps Script as a **Web App** that accepts POST requests and appends rows to your sheet.
2. Open the app's **Settings** page.
3. Paste the Web App URL, click **Save URL**, then **Test connection**.

## 📁 Project Structure

```
MadPotatoSystem/
├── app.py                  # Flask backend and API routes
├── index.html              # Single-page frontend
├── train_nlp.py            # Intent classifier training
├── train_prophet.py        # Prophet training and feature engineering
├── evaluate_nlp.py         # Classifier evaluation
├── evaluate_forecast.py    # Rolling backtest vs baselines
├── sales_data.csv          # Daily operational dataset
├── nlp_pipeline.joblib     # Trained intent classifier
├── prophet_model.joblib    # Trained Prophet model
├── forecast_metrics.json   # Backtest results
├── nlp_metrics.json        # Classifier results
├── requirements.txt
└── README.md
```

> Adjust this tree if your file names differ.

## ⚠️ Known Limitations

- No login or authentication, and the app runs in debug mode.
- The logbook is held in memory with CSV saves, which suits a single store only.
- Payday and holiday effects are not yet modeled in the deployed forecast.
- Forecast accuracy is currently below a 7-day moving average baseline.

## 🗺️ Roadmap

- [ ] Tune Prophet (lower `changepoint_prior_scale`, additive seasonality, payday and holiday regressors)
- [ ] Compare SARIMAX, XGBoost, and LightGBM on the same backtest
- [ ] Add user login and a database (SQLite or PostgreSQL) before multi-branch support
- [ ] Usability testing with store staff (task time and errors vs the paper logbook)

## 👤 Author

**George Albin M. Seda**
Computer Engineering student, Technological Institute of the Philippines, Quezon City
GitHub: [@geoseda](https://github.com/geoseda)

## 🙏 Acknowledgments

- **Val Aric Bagongon**, franchisee of the Mad Potato Teresa branch, for serving as the client.
- Project documentation was prepared together with groupmates Kerwin Jan Catungal, Keon Martin Llorin, and Dave Adriane Madriago.

## 📄 License

This project was developed for academic purposes. Add a license (for example MIT) if you want others to reuse the code.

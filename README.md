# Apple Stock Price Forecasting

A Streamlit web app that forecasts Apple (AAPL) stock prices using **ARIMA**, **ARIMAX**, and **XGBoost** models.

## What This App Does

| Section | Model | What it predicts |
|---|---|---|
| ARIMA Forecast | ARIMA (1,1,1) | Future AAPL price using past prices only |
| ARIMAX Forecast | ARIMAX (1,1,1) | Future AAPL price using AAPL + TXN prices |
| XGBoost Classifier | XGBoostClassifier | Will price go UP or DOWN tomorrow? |

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/apple-stock-forecaster.git
cd apple-stock-forecaster
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Make sure AAPL.csv is in the same folder, then run
```bash
streamlit run app.py
```

## Project Structure

```
apple_stock_app/
├── app.py              ← Streamlit web app (main file)
├── model.py            ← All ML model logic (ARIMA, ARIMAX, XGBoost)
├── utils.py            ← Visualization and helper functions
├── AAPL.csv            ← Dataset (Apple + TXN monthly prices)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

## Dependencies

- Python 3.8+
- streamlit, pandas, numpy, matplotlib, seaborn
- statsmodels (ARIMA/ARIMAX)
- scikit-learn (evaluation metrics)
- xgboost (XGBoost classifier)
- yfinance (live data download)

## Dataset

`AAPL.csv` contains monthly stock prices for:
- **AAPL** — Apple Inc. (traded on NASDAQ)
- **TXN** — Texas Instruments (major chip supplier to Apple)

"""
app.py - Streamlit Web App for Apple Stock Price Forecasting
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from model import (
    load_and_prepare_data,
    get_univariate_series,
    decompose_series,
    train_arima,
    forecast_arima,
    train_arimax,
    prepare_xgboost_data,
    train_xgboost,
    predict_xgboost,
)
from utils import (
    plot_stock_price,
    plot_decomposition,
    plot_arima_forecast,
    plot_xgboost_predictions,
    build_forecast_dataframe,
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Apple Stock Forecaster",
    page_icon="🍎",
    layout="wide"
)

st.title("Apple Stock Price Forecasting")
st.markdown("This app uses **ARIMA**, **ARIMAX**, and **XGBoost** models to analyze and forecast Apple stock prices.")

# ── Load Data ────────────────────────────────────────────────
DATA_PATH = "AAPL.csv"

@st.cache_data
def load_data():
    return load_and_prepare_data(DATA_PATH)

try:
    data = load_data()
except Exception as e:
    st.error(f"Could not load AAPL.csv. Make sure it's in the same folder as app.py.\nError: {e}")
    st.stop()

# ── Sidebar Navigation ───────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section:", [
    "Data Overview",
    "ARIMA Forecast (Univariate)",
    "ARIMAX Forecast (Bivariate)",
    "XGBoost Classifier"
])

# ════════════════════════════════════════════════════════════
# SECTION 1 — Data Overview
# ════════════════════════════════════════════════════════════
if section == "Data Overview":
    st.header("Data Overview")
    st.markdown("Raw Apple stock data loaded from **AAPL.csv**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Date Range", f"{data['Date'].min().date()} → {data['Date'].max().date()}")

    st.subheader("First 5 rows")
    st.dataframe(data.head())

    st.subheader("Historical AAPL Stock Price")
    df = get_univariate_series(data)
    fig = plot_stock_price(df)
    st.pyplot(fig)

    st.subheader("Time Series Decomposition")
    st.markdown("Breaking the price into **Trend**, **Seasonal pattern**, and **Residual noise**.")
    try:
        trend, seasonal, residual = decompose_series(df)
        fig2 = plot_decomposition(df, trend, seasonal, residual)
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Decomposition requires at least 2 full cycles of data. Error: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 2 — ARIMA Forecast
# ════════════════════════════════════════════════════════════
elif section == "ARIMA Forecast (Univariate)":
    st.header("ARIMA Forecast — Using Only AAPL Price")
    st.markdown("""
    **ARIMA** (Auto Regressive Integrated Moving Average) forecasts future prices 
    using only past AAPL prices. We use order **(p=1, d=1, q=1)**.
    """)

    p = st.sidebar.slider("p (AR term)", 0, 3, 1)
    d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
    q = st.sidebar.slider("q (MA term)", 0, 3, 1)

    if st.button("Train ARIMA & Forecast"):
        with st.spinner("Training ARIMA model... this may take a moment ⏳"):
            try:
                df = get_univariate_series(data)
                ar_model = train_arima(df, order=(p, d, q))
                ypred, conf_int = forecast_arima(ar_model, steps=2)

                st.subheader("Model Summary")
                st.text(str(ar_model.summary()))

                dp, lower_int, upper_int = build_forecast_dataframe(ypred, conf_int)

                st.subheader("Forecast Results")
                st.dataframe(dp)

                data_indexed = data.copy().set_index('Date') if 'Date' in data.columns else data
                fig = plot_arima_forecast(data_indexed, dp, lower_int, upper_int)
                st.pyplot(fig)

                from sklearn.metrics import mean_absolute_error
                mae = mean_absolute_error(dp['price_actual'].astype(float), dp['price_predicted'].astype(float))
                st.success(f"ARIMA Mean Absolute Error (MAE): **${mae:.2f}**")

            except Exception as e:
                st.error(f"ARIMA training failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 3 — ARIMAX Forecast
# ════════════════════════════════════════════════════════════
elif section == "ARIMAX Forecast (Bivariate)":
    st.header("ARIMAX Forecast — Using AAPL + TXN")
    st.markdown("""
    **ARIMAX** is ARIMA with an **exogenous variable** — an extra input from outside the series.
    Here we use **TXN (Texas Instruments)** stock price as the extra input, since TXN supplies 
    chips to Apple.
    """)

    if 'TXN' not in data.columns:
        st.error("TXN column not found in your CSV. Make sure AAPL.csv has a TXN column.")
    else:
        if st.button("Train ARIMAX & Forecast"):
            with st.spinner("Training ARIMAX model..."):
                try:
                    dfx = data.copy()
                    if 'Date' in dfx.columns:
                        dfx = dfx.set_index('Date')
                    dfx = dfx.iloc[0:-2, 0:3]

                    arimax = train_arimax(dfx, order=(1, 1, 1))

                    ex = data['TXN'].iloc[-2:].values
                    forecast = arimax.get_forecast(2, exog=ex)
                    ypred = forecast.predicted_mean
                    conf_int = forecast.conf_int(alpha=0.05)

                    st.subheader("Model Summary")
                    st.text(str(arimax.summary()))

                    dp, lower_int, upper_int = build_forecast_dataframe(ypred, conf_int)

                    st.subheader("Forecast Results")
                    st.dataframe(dp)

                    data_indexed = data.set_index('Date') if 'Date' in data.columns else data
                    fig = plot_arima_forecast(data_indexed, dp, lower_int, upper_int)
                    st.pyplot(fig)

                    from sklearn.metrics import mean_absolute_error
                    mae = mean_absolute_error(dp['price_actual'].astype(float), dp['price_predicted'].astype(float))
                    st.success(f"ARIMAX Mean Absolute Error (MAE): **${mae:.2f}**")

                except Exception as e:
                    st.error(f"ARIMAX training failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 4 — XGBoost Classifier
# ════════════════════════════════════════════════════════════
elif section == "XGBoost Classifier":
    st.header("XGBoost — Will the Price Go Up or Down Tomorrow?")
    st.markdown("""
    Instead of predicting exact prices, **XGBoost** classifies whether the stock 
    will go **UP (1)** or **DOWN (0)** the next day. It uses Open, High, Low, Close, 
    and Volume as features.
    """)

    st.info("⚠️ This section uses **yfinance** to download live AAPL data. Make sure `yfinance` is installed.")

    if st.button("Train XGBoost & Evaluate"):
        with st.spinner("Downloading live data & training XGBoost... ⏳"):
            try:
                import yfinance as yf
                live_data = yf.download("AAPL", start="2000-01-01", end="2025-05-31")
                live_data.columns = live_data.columns.get_level_values(0)

                train, test, features = prepare_xgboost_data(live_data)
                model = train_xgboost(train, features)
                results, score = predict_xgboost(model, train, test, features)

                col1, col2 = st.columns(2)
                col1.metric("Precision Score", f"{score:.2%}")
                col2.metric("Test Days", len(test))

                st.subheader("Predictions vs Actual")
                st.dataframe(results.tail(10))

                fig = plot_xgboost_predictions(results)
                st.pyplot(fig)

                st.success(f"XGBoost Precision: **{score:.2%}** — A score above 50% means it's better than random guessing!")

            except ImportError:
                st.error("yfinance not installed. Add 'yfinance' to your requirements.txt")
            except Exception as e:
                st.error(f"XGBoost failed: {e}")

# ── Footer ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Individual Project**")
st.sidebar.markdown("Apple Stock Price Forecasting using ARIMA & XGBoost")

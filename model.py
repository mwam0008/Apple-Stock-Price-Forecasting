"""
model.py - All ML model logic for Apple Stock Forecasting
"""

import logging
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, precision_score
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load CSV and convert Date column to datetime."""
    try:
        logging.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'])
        logging.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def get_univariate_series(data: pd.DataFrame) -> pd.DataFrame:
    """Extract just the AAPL price series with Date as index."""
    try:
        df = data.iloc[:-2, 0:2].copy()
        df = df.set_index('Date')
        logging.info("Univariate series prepared.")
        return df
    except Exception as e:
        logging.error(f"Error preparing univariate series: {e}")
        raise


def decompose_series(df: pd.DataFrame):
    """Break the time series into trend, seasonal, and residual parts."""
    try:
        logging.info("Decomposing time series...")
        decomposed = seasonal_decompose(df['AAPL'])
        return decomposed.trend, decomposed.seasonal, decomposed.resid
    except Exception as e:
        logging.error(f"Decomposition failed: {e}")
        raise


def train_arima(df: pd.DataFrame, order=(1, 1, 1)):
    """Train an ARIMA model on the AAPL price series."""
    try:
        logging.info(f"Training ARIMA{order} model...")
        arima = ARIMA(df['AAPL'], order=order)
        ar_model = arima.fit()
        logging.info("ARIMA model trained successfully.")
        return ar_model
    except Exception as e:
        logging.error(f"ARIMA training failed: {e}")
        raise


def forecast_arima(ar_model, steps: int = 2):
    """Generate forecast from a trained ARIMA model."""
    try:
        logging.info(f"Forecasting {steps} steps ahead...")
        forecast = ar_model.get_forecast(steps)
        ypred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        return ypred, conf_int
    except Exception as e:
        logging.error(f"Forecasting failed: {e}")
        raise


def train_arimax(dfx: pd.DataFrame, order=(1, 1, 1)):
    """Train an ARIMAX model using TXN as an exogenous variable."""
    try:
        logging.info(f"Training ARIMAX{order} model with TXN as exogenous variable...")
        model = ARIMA(dfx['AAPL'], exog=dfx['TXN'], order=order)
        arimax = model.fit()
        logging.info("ARIMAX model trained successfully.")
        return arimax
    except Exception as e:
        logging.error(f"ARIMAX training failed: {e}")
        raise


def prepare_xgboost_data(data: pd.DataFrame):
    """Prepare data for XGBoost classification (predict if price goes up or down)."""
    try:
        logging.info("Preparing data for XGBoost...")
        data = data.copy()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        train = data.iloc[:-30]
        test = data.iloc[-30:]
        logging.info(f"Train size: {train.shape}, Test size: {test.shape}")
        return train, test, features
    except Exception as e:
        logging.error(f"XGBoost data prep failed: {e}")
        raise


def train_xgboost(train: pd.DataFrame, features: list):
    """Train an XGBoost classifier."""
    try:
        logging.info("Training XGBoost model...")
        model = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
        model.fit(train[features], train['Target'])
        logging.info("XGBoost model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"XGBoost training failed: {e}")
        raise


def predict_xgboost(model, train: pd.DataFrame, test: pd.DataFrame, features: list):
    """Make predictions with a trained XGBoost model."""
    try:
        model.fit(train[features], train['Target'])
        preds = model.predict(test[features])
        preds = pd.Series(preds, index=test.index, name='predictions')
        combined = pd.concat([test['Target'], preds], axis=1)
        score = precision_score(test['Target'], preds)
        logging.info(f"XGBoost Precision Score: {score:.4f}")
        return combined, score
    except Exception as e:
        logging.error(f"XGBoost prediction failed: {e}")
        raise


def backtest(data: pd.DataFrame, model, features: list, start=5031, step=120):
    """Run backtesting to simulate real-world model performance."""
    try:
        logging.info("Running backtest...")
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[:i].copy()
            test = data.iloc[i:(i + step)].copy()
            model.fit(train[features], train['Target'])
            preds = model.predict(test[features])
            preds = pd.Series(preds, index=test.index, name='predictions')
            combined = pd.concat([test['Target'], preds], axis=1)
            all_predictions.append(combined)
        result = pd.concat(all_predictions)
        score = precision_score(result['Target'], result['predictions'])
        logging.info(f"Backtest Precision Score: {score:.4f}")
        return result, score
    except Exception as e:
        logging.error(f"Backtesting failed: {e}")
        raise

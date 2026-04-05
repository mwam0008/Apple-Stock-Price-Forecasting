"""
utils.py - Helper functions for visualization and data display
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # needed for Streamlit (no display screen)

logging.basicConfig(level=logging.INFO)


def plot_stock_price(df: pd.DataFrame):
    """Plot historical AAPL stock price."""
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['AAPL'], color='steelblue')
        ax.set_title('Apple (AAPL) Stock Price Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting stock price: {e}")
        raise


def plot_decomposition(df: pd.DataFrame, trend, seasonal, residual):
    """Plot trend, seasonal, and residual components."""
    try:
        fig, axes = plt.subplots(4, 1, figsize=(12, 8))
        axes[0].plot(df['AAPL'], color='black', label='Original')
        axes[0].legend(loc='upper left')
        axes[1].plot(trend, color='red', label='Trend')
        axes[1].legend(loc='upper left')
        axes[2].plot(seasonal, color='blue', label='Seasonal')
        axes[2].legend(loc='upper left')
        axes[3].plot(residual, color='black', label='Residual')
        axes[3].legend(loc='upper left')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting decomposition: {e}")
        raise


def plot_arima_forecast(data: pd.DataFrame, dp: pd.DataFrame, lower_int, upper_int):
    """Plot ARIMA forecast vs actual price."""
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data['AAPL'], label='Actual', color='steelblue')
        ax.plot(dp['price_predicted'].astype(float), color='orange', label='Prediction')
        ax.fill_between(dp.index,
                        lower_int.astype(float),
                        upper_int.astype(float),
                        color='gray', alpha=0.3, label='Confidence Interval')
        ax.set_title('ARIMA Model - Forecast vs Actual')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting ARIMA forecast: {e}")
        raise


def plot_xgboost_predictions(test_df):
    """Plot XGBoost predicted direction vs actual direction."""
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(test_df['Target'].values, label='Actual Direction', color='steelblue')
        ax.plot(test_df['predictions'].values, label='Predicted Direction', color='orange', alpha=0.7)
        ax.set_title('XGBoost: Predicted vs Actual (1=Price Up, 0=Price Down)')
        ax.set_xlabel('Days')
        ax.set_ylabel('Direction')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting XGBoost predictions: {e}")
        raise


def build_forecast_dataframe(ypred, conf_int, col_name='AAPL'):
    """Build a clean dataframe from ARIMA forecast output."""
    try:
        dates = pd.Series(['2024-01-01', '2024-02-01'])
        price_actual = pd.Series(['184.40', '185.04'])
        price_predicted = pd.Series(ypred.values)
        lower_int = pd.Series(conf_int[f'lower {col_name}'].values)
        upper_int = pd.Series(conf_int[f'upper {col_name}'].values)

        dp = pd.DataFrame(
            [dates, price_actual, lower_int, price_predicted, upper_int],
            index=['Date', 'price_actual', 'lower_int', 'price_predicted', 'upper_int']
        ).T
        dp = dp.set_index('Date')
        dp.index = pd.to_datetime(dp.index)
        return dp, lower_int, upper_int
    except Exception as e:
        logging.error(f"Error building forecast dataframe: {e}")
        raise

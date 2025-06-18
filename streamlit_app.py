import streamlit as st
import numpy as np
import pandas as pd

from utils.data_loader import get_combined_stock_data
from utils.preprocessing import calculate_returns, normalize_weights
from utils.calculations import historical_var, parametric_var, monte_carlo_var

st.set_page_config(page_title="VaR Calculator", layout="centered")

st.title("📉 Value at Risk (VaR) Calculator")

# Inputs
tickers_input = st.text_input("Enter stock tickers (comma-separated, max 10)", "AAPL,TSLA,GOOGL")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

if len(tickers) > 10:
    st.warning("Please enter 10 or fewer tickers.")
    st.stop()

weights_input = st.text_input("Enter portfolio weights (comma-separated)", "0.4,0.4,0.2")

try:
    weights = [float(w.strip()) for w in weights_input.split(',')]
    weights = normalize_weights(weights)
    if len(weights) != len(tickers):
        st.warning("Number of weights must match number of tickers.")
        st.stop()
except Exception as e:
    st.warning(f"Weight input error: {e}")
    st.stop()

confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
method = st.selectbox("Select VaR Method", ["Historical", "Variance-Covariance", "Monte Carlo"])
period_days = st.slider("Number of past days for data", 60, 365, 120)
simulations = st.number_input("Monte Carlo simulations", 1000, 50000, 10000, step=1000)
horizon = st.number_input("Time horizon (in days)", 1, 30, 1)

# Compute VaR
if st.button("Calculate VaR"):

    try:
        st.info("Fetching and processing stock data...")
        combined_df, failed_tickers = get_combined_stock_data(tickers, period_days=period_days)

        if failed_tickers:
            st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")

        pct_returns, log_returns = calculate_returns(combined_df)

        if method == "Historical":
            var = historical_var(pct_returns, confidence_level=confidence, weights=weights)
        elif method == "Variance-Covariance":
            var = parametric_var(log_returns, confidence_level=confidence, weights=weights)
        elif method == "Monte Carlo":
            var = monte_carlo_var(log_returns, confidence_level=confidence,
                                  weights=weights, num_simulations=simulations, time_horizon=horizon)

        st.success(f"{method} VaR at {int(confidence*100)}% confidence: {var:.4f}")

    except ValueError as e:
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")

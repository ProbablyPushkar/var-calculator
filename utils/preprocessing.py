import numpy as np
import pandas as pd

def calculate_returns(df_prices):
    """
    Calculates both percentage and log returns from price DataFrame.
    Assumes 'Date' is the first column.
    """
    prices = df_prices.iloc[:, 1:]

    pct_returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Add back the Date column
    pct_returns.insert(0, 'Date', df_prices['Date'].iloc[1:])
    log_returns.insert(0, 'Date', df_prices['Date'].iloc[1:])

    return pct_returns, log_returns


def normalize_weights(weights):
    """
    Ensures portfolio weights sum to 1.
    """
    weights = np.array(weights, dtype=float)
    if np.sum(weights) == 0:
        raise ValueError("Sum of weights cannot be zero.")
    return weights / np.sum(weights)


def align_and_clean(df):
    """
    Placeholder for future cleaning logic — useful if you start working with 
    multiple data sources or time zones, or want to drop certain dates.
    """
    df = df.copy()
    df.dropna(inplace=True)
    return df

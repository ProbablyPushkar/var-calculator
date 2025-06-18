import numpy as np
from scipy.stats import norm

def historical_var(returns_df, confidence_level=0.95, weights=None):
    """
    Calculates Historical VaR.
    - returns_df: % returns, expects first column to be Date.
    - weights: list or np.array, optional. If None, equal weights are used.
    """
    returns = returns_df.iloc[:, 1:]

    if weights is None:
        weights = np.array([1 / returns.shape[1]] * returns.shape[1])
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)

    portfolio_returns = returns.dot(weights)
    var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    return var


def parametric_var(log_returns_df, confidence_level=0.95, weights=None):
    """
    Calculates Parametric VaR assuming normal distribution.
    - log_returns_df: log returns, first column is Date.
    """
    returns = log_returns_df.iloc[:, 1:]

    if weights is None:
        weights = np.array([1 / returns.shape[1]] * returns.shape[1])
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)

    portfolio_mean = np.dot(weights, returns.mean())
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))

    z_score = norm.ppf(1 - confidence_level)
    var = -(portfolio_mean + z_score * portfolio_std)

    return var

def monte_carlo_var(log_returns_df, confidence_level=0.95, weights=None, num_simulations=10000, time_horizon=1):
    """
    Monte Carlo Simulation for Value at Risk.

    Parameters:
    - log_returns_df: DataFrame of log returns (first column is 'Date').
    - confidence_level: Confidence level for VaR.
    - weights: Portfolio weights, normalized automatically if not.
    - num_simulations: Number of simulations (default: 10,000).
    - time_horizon: Number of days to project into the future.
    """

    returns = log_returns_df.iloc[:, 1:]  # Remove Date
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    if weights is None:
        weights = np.array([1 / returns.shape[1]] * returns.shape[1])
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)

    # Cholesky decomposition for correlated random normal generation
    L = np.linalg.cholesky(cov_matrix)

    # Simulate correlated returns
    np.random.seed(42)  # For reproducibility
    random_normals = np.random.normal(size=(time_horizon, len(returns.columns), num_simulations))
    correlated_returns = np.matmul(L, random_normals)

    # Projected portfolio return paths
    portfolio_returns = np.dot(weights, mean_returns.values.reshape(-1, 1)) * time_horizon + \
                        np.dot(weights, correlated_returns.sum(axis=0))

    # Calculate VaR
    var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    return var

def print_var_results(pct_returns, log_returns, weights=None, confidence=0.95, simulations=10000, horizon=1):
    hist_var = historical_var(pct_returns, confidence_level=confidence, weights=weights)
    param_var = parametric_var(log_returns, confidence_level=confidence, weights=weights)
    mc_var = monte_carlo_var(log_returns, confidence_level=confidence, weights=weights,
                              num_simulations=simulations, time_horizon=horizon)

    print(f"Historical VaR     ({int(confidence * 100)}%): {hist_var:.4f}")
    print(f"Parametric VaR     ({int(confidence * 100)}%): {param_var:.4f}")
    print(f"Monte Carlo VaR    ({int(confidence * 100)}%, {simulations} simulations): {mc_var:.4f}")
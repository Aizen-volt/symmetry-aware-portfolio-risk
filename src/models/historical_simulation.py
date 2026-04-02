"""
Historical Simulation VaR/ES baseline.

Non-parametric: uses rolling empirical quantile of portfolio returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_simulation_var_es(
    portfolio_returns: pd.Series,
    alpha: float = 0.05,
    window: int = 250,
    min_periods: int = 60,
) -> pd.DataFrame:
    """
    Compute one-day-ahead VaR and ES via historical simulation.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns (already weighted).
    alpha : float
        Tail level (e.g. 0.05 or 0.01).
    window : int
        Rolling window size.
    min_periods : int
        Minimum observations required.

    Returns
    -------
    pd.DataFrame with columns ['var', 'es'] indexed by date.
    """
    var_list = []
    es_list = []
    dates = []

    returns = portfolio_returns.values
    index = portfolio_returns.index

    for t in range(min_periods, len(returns)):
        start = max(0, t - window)
        hist = returns[start:t]
        hist = hist[~np.isnan(hist)]

        if len(hist) < min_periods:
            var_list.append(np.nan)
            es_list.append(np.nan)
        else:
            q = np.quantile(hist, alpha)
            tail = hist[hist <= q]
            if len(tail) == 0:
                es_val = q
            else:
                es_val = float(np.mean(tail))
            var_list.append(float(q))
            es_list.append(es_val)

        dates.append(index[t])

    return pd.DataFrame({
        "var": var_list,
        "es": es_list,
    }, index=dates)


def evaluate_historical_simulation(
    portfolio_returns: pd.Series,
    alpha: float,
    window: int = 250,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    """
    Run historical simulation and evaluate on the test split
    (matching the neural model splits).
    """
    hs = historical_simulation_var_es(
        portfolio_returns, alpha=alpha, window=window, min_periods=60,
    )

    # Drop duplicate date indices (Fama-French data can have them)
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep="first")]

    # Align with portfolio returns (HS is one-step-ahead, so var[t] forecasts r[t])
    aligned = pd.DataFrame({
        "y": portfolio_returns,
        "var": hs["var"],
        "es": hs["es"],
    }).dropna()

    n = len(aligned)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    test_data = aligned.iloc[n_train + n_val:]

    if len(test_data) == 0:
        return {"error": "No test data"}

    y = test_data["y"].values
    var = test_data["var"].values
    es = test_data["es"].values

    cov_rate = float(np.mean(y < var))
    cov_err = float(abs(cov_rate - alpha))

    hits = y < var
    if hits.sum() > 0:
        avg_exc = float(np.mean(var[hits] - y[hits]))
    else:
        avg_exc = 0.0

    return {
        "coverage_rate": cov_rate,
        "coverage_error": cov_err,
        "avg_exceedance_loss": avg_exc,
        "mean_var": float(np.mean(var)),
        "mean_es": float(np.mean(es)),
        "n_test": len(test_data),
        "n_total": n,
    }
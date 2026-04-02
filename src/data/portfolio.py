from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd

PortfolioName = Literal["equal_weight", "vol_scaled"]


def make_equal_weight_portfolio(returns_df: pd.DataFrame) -> pd.DataFrame:
    valid = returns_df.notna().astype(float)
    denom = valid.sum(axis=1).replace(0.0, np.nan)
    return valid.div(denom, axis=0).fillna(0.0)


def make_vol_scaled_portfolio(
    returns_df: pd.DataFrame,
    vol_lookback: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-8,
) -> pd.DataFrame:
    if min_periods is None:
        min_periods = max(5, vol_lookback // 2)

    rolling_vol = returns_df.rolling(window=vol_lookback, min_periods=min_periods).std()
    inv_vol = 1.0 / (rolling_vol + eps)
    inv_vol = inv_vol.where(returns_df.notna(), 0.0)

    denom = inv_vol.abs().sum(axis=1).replace(0.0, np.nan)
    return inv_vol.div(denom, axis=0).fillna(0.0)


def make_portfolio_weights(
    returns_df: pd.DataFrame,
    portfolio: PortfolioName,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    if portfolio == "equal_weight":
        return make_equal_weight_portfolio(returns_df)
    if portfolio == "vol_scaled":
        return make_vol_scaled_portfolio(returns_df, vol_lookback=vol_lookback)
    raise ValueError(f"Unknown portfolio: {portfolio}")


def compute_portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.Series:
    """
    Contemporaneous portfolio return:
        r_p,t = sum_i w_i,t * r_i,t

    Useful for descriptive analysis, but not fully aligned with the
    one-step-ahead forecasting task used by the neural models.
    """
    aligned_returns = returns_df.reindex_like(weights_df)
    rp = (aligned_returns.fillna(0.0) * weights_df.fillna(0.0)).sum(axis=1)
    return rp.rename("portfolio_return")


def compute_next_day_portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """
    One-step-ahead portfolio return series aligned with the forecasting task:
        r_p,t+1 = sum_i w_i,t * r_i,t+1

    The returned series is indexed by time t (decision date / forecast origin),
    not by time t+1. This matches the neural dataset setup, where features and
    weights are observed at time t and the target is the next-day portfolio return.
    """
    if horizon != 1:
        raise ValueError("Current implementation supports horizon=1 only.")

    future_returns = returns_df.shift(-horizon)
    aligned_future_returns = future_returns.reindex_like(weights_df)

    rp_next = (aligned_future_returns.fillna(0.0) * weights_df.fillna(0.0)).sum(axis=1)

    # Last rows do not have t+1 returns available after shifting.
    valid_mask = future_returns.notna().any(axis=1)
    rp_next = rp_next.loc[valid_mask]

    return rp_next.rename("portfolio_return_t_plus_1")
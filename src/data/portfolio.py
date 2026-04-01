from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


PortfolioName = Literal["equal_weight", "vol_scaled"]


def make_equal_weight_portfolio(returns_df: pd.DataFrame) -> pd.DataFrame:
    valid = returns_df.notna().astype(float)
    denom = valid.sum(axis=1).replace(0.0, np.nan)
    w = valid.div(denom, axis=0).fillna(0.0)
    return w


def make_vol_scaled_portfolio(
    returns_df: pd.DataFrame,
    vol_lookback: int = 20,
    min_periods: int | None = None,
    eps: float = 1e-8,
) -> pd.DataFrame:
    if min_periods is None:
        min_periods = max(5, vol_lookback // 2)

    rolling_vol = returns_df.rolling(
        window=vol_lookback,
        min_periods=min_periods,
    ).std()

    inv_vol = 1.0 / (rolling_vol + eps)
    inv_vol = inv_vol.where(returns_df.notna(), 0.0)

    denom = inv_vol.abs().sum(axis=1).replace(0.0, np.nan)
    w = inv_vol.div(denom, axis=0).fillna(0.0)
    return w


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
    aligned_returns = returns_df.reindex_like(weights_df)
    rp = (aligned_returns.fillna(0.0) * weights_df.fillna(0.0)).sum(axis=1)
    return rp.rename("portfolio_return")
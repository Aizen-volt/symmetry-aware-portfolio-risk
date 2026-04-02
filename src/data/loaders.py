from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


DatasetName = Literal["industry12", "industry49"]

DATASET_FILE_MAP = {
    "industry12": "ff_12_industry_daily.csv",
    "industry49": "ff_49_industry_daily.csv",
}


def _coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if is_datetime64_any_dtype(idx):
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out.sort_index()
    idx_str = pd.Index(idx).astype(str).str.strip()
    parsed = None
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            parsed = pd.to_datetime(idx_str, format=fmt, errors="raise")
            break
        except Exception:
            continue
    if parsed is None:
        parsed = pd.to_datetime(idx_str, errors="coerce")
    out = df.copy()
    out.index = parsed
    out = out.loc[~out.index.isna()].sort_index()
    return out


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    return df[keep_cols].copy()


def _replace_missing_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([-99.99, -999, -999.0], np.nan)
    return out


def _maybe_convert_percent_to_decimal(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    finite = finite[np.abs(finite) > 0]
    if finite.size == 0:
        return df
    med_abs = np.median(np.abs(finite))
    if med_abs > 0.2:
        return df / 100.0
    return df


def load_ff_industry_returns(
    dataset: DatasetName,
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    if dataset not in DATASET_FILE_MAP:
        raise ValueError(f"Unknown dataset: {dataset}")
    path = Path(data_dir) / DATASET_FILE_MAP[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path, index_col=0, skipinitialspace=True, engine="python")
    df = _drop_unnamed_columns(df)
    df = _coerce_datetime_index(df)
    df = _coerce_numeric(df)
    df = _replace_missing_sentinels(df)
    df = _maybe_convert_percent_to_decimal(df)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
    return df


def infer_market_series(returns_df: pd.DataFrame) -> pd.Series:
    return returns_df.mean(axis=1).rename("market_return")

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiAssetPanelDataset(Dataset):
    def __init__(self, returns_df, market_series=None, weights_df=None, lookback=60, horizon=1, min_assets=5, vol_lookback=20, normalize_inputs=False):
        super().__init__()
        if horizon != 1:
            raise ValueError("Current implementation supports horizon=1 only.")
        self.returns_df = returns_df.sort_index().copy()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.min_assets = int(min_assets)
        self.vol_lookback = int(vol_lookback)
        self.normalize_inputs = bool(normalize_inputs)
        if market_series is None:
            market_series = self.returns_df.mean(axis=1)
        self.market_series = market_series.reindex(self.returns_df.index).copy()
        if weights_df is None:
            valid = self.returns_df.notna().astype(float)
            denom = valid.sum(axis=1).replace(0.0, np.nan)
            weights_df = valid.div(denom, axis=0).fillna(0.0)
        else:
            weights_df = weights_df.reindex_like(self.returns_df).fillna(0.0)
        self.weights_df = weights_df
        self.assets = list(self.returns_df.columns)
        self.n_assets = len(self.assets)
        self.feature_mean = None
        self.feature_std = None
        self._ret = self.returns_df.to_numpy(dtype=np.float32)
        self._mkt = self.market_series.to_numpy(dtype=np.float32)
        self._wgt = self.weights_df.to_numpy(dtype=np.float32)
        self._vol = self._rolling_std(self.returns_df, self.vol_lookback).to_numpy(dtype=np.float32)
        self._downside_vol = self._rolling_downside_std(self.returns_df, self.vol_lookback).to_numpy(dtype=np.float32)
        self._rolling_mean = self.returns_df.rolling(window=self.vol_lookback, min_periods=max(5, self.vol_lookback // 2)).mean().to_numpy(dtype=np.float32)
        self.valid_timestamps = self._build_valid_index()

    @staticmethod
    def _rolling_std(df, window):
        return df.rolling(window=window, min_periods=max(5, window // 2)).std()
    @staticmethod
    def _rolling_downside_std(df, window):
        neg = df.where(df < 0.0, 0.0)
        return neg.rolling(window=window, min_periods=max(5, window // 2)).std()

    def _build_valid_index(self):
        valid = []
        T = len(self.returns_df)
        for t in range(self.lookback - 1, T - self.horizon):
            x_window = self._ret[t - self.lookback + 1:t + 1]
            n_valid_assets = np.sum(~np.all(np.isnan(x_window), axis=0))
            if n_valid_assets < self.min_assets:
                continue
            y_assets = self._ret[t + self.horizon]
            w_row = self._wgt[t]
            valid_target_mask = (~np.isnan(y_assets)) & (w_row != 0.0)
            if valid_target_mask.sum() == 0:
                continue
            valid.append(t)
        return valid

    def __len__(self):
        return len(self.valid_timestamps)

    def fit_normalizer(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        chunks = []
        prev_flag = self.normalize_inputs
        self.normalize_inputs = False
        for idx in indices:
            sample = self[idx]
            x = sample["x"].numpy()
            mask = sample["asset_mask"].numpy().astype(bool)
            if mask.sum() == 0:
                continue
            chunks.append(x[mask].reshape(-1, x.shape[-1]))
        self.normalize_inputs = prev_flag
        if not chunks:
            raise RuntimeError("No valid data available to fit normalizer.")
        cat = np.concatenate(chunks, axis=0)
        mean = np.nanmean(cat, axis=0)
        std = np.nanstd(cat, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = std.astype(np.float32)

    def set_normalizer(self, mean, std):
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def _make_features(self, t):
        sl = slice(t - self.lookback + 1, t + 1)
        ret = self._ret[sl]; mkt = self._mkt[sl][:, None]
        vol = self._vol[sl]; dvol = self._downside_vol[sl]; rmean = self._rolling_mean[sl]
        L, N = ret.shape
        mkt_b = np.repeat(mkt, N, axis=1)
        feat = np.stack([ret, mkt_b, vol, dvol, rmean], axis=-1)
        feat = np.transpose(feat, (1, 0, 2))
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize_inputs:
            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("Normalizer not fitted.")
            feat = (feat - self.feature_mean[None, None, :]) / self.feature_std[None, None, :]
        return feat.astype(np.float32)

    def _make_asset_mask(self, t):
        sl = slice(t - self.lookback + 1, t + 1)
        x_window = self._ret[sl]
        mask = ~np.all(np.isnan(x_window), axis=0)
        return mask.astype(np.float32)

    def _make_weights(self, t, asset_mask):
        w = self._wgt[t].copy()
        w = np.where(asset_mask > 0.0, w, 0.0)
        s = float(np.sum(np.abs(w)))
        if s < 1e-12:
            n_valid = int(asset_mask.sum())
            if n_valid == 0:
                return np.zeros_like(w, dtype=np.float32)
            w = asset_mask / asset_mask.sum()
        else:
            w = w / s
        return w.astype(np.float32)

    def _make_target(self, t, weights):
        y_assets = self._ret[t + self.horizon].copy()
        y_assets = np.nan_to_num(y_assets, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.sum(weights * y_assets))

    def __getitem__(self, idx):
        t = self.valid_timestamps[idx]
        x = self._make_features(t)
        asset_mask = self._make_asset_mask(t)
        weights = self._make_weights(t, asset_mask)
        y = self._make_target(t, weights)
        return {
            "x": torch.from_numpy(x),
            "asset_mask": torch.from_numpy(asset_mask),
            "weights": torch.from_numpy(weights),
            "y": torch.tensor(y, dtype=torch.float32),
        }

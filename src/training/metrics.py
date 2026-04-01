from __future__ import annotations

import numpy as np


def coverage_rate(y: np.ndarray, var: np.ndarray) -> float:
    return float(np.mean(y < var))


def coverage_error(y: np.ndarray, var: np.ndarray, alpha: float) -> float:
    return float(abs(coverage_rate(y, var) - alpha))


def avg_exceedance_loss(y: np.ndarray, var: np.ndarray) -> float:
    hits = y < var
    if hits.sum() == 0:
        return 0.0
    return float(np.mean(var[hits] - y[hits]))


def basic_tail_metrics(y: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> dict[str, float]:
    return {
        "coverage_rate": coverage_rate(y, var),
        "coverage_error": coverage_error(y, var, alpha),
        "avg_exceedance_loss": avg_exceedance_loss(y, var),
        "mean_var": float(np.mean(var)),
        "mean_es": float(np.mean(es)),
        "mean_y": float(np.mean(y)),
        "std_y": float(np.std(y)),
    }


def permutation_prediction_drift(preds: np.ndarray) -> dict[str, float]:
    centered = preds - preds[:, :1]
    abs_drift = np.abs(centered)
    return {
        "mean_abs_drift": float(abs_drift.mean()),
        "median_abs_drift": float(np.median(abs_drift)),
        "max_abs_drift": float(abs_drift.max()),
    }
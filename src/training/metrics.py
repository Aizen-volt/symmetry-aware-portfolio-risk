"""
Evaluation metrics for tail-risk forecasting.
Includes Christoffersen unconditional and conditional coverage tests.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def coverage_rate(y: np.ndarray, var: np.ndarray) -> float:
    return float(np.mean(y < var))


def coverage_error(y: np.ndarray, var: np.ndarray, alpha: float) -> float:
    return float(abs(coverage_rate(y, var) - alpha))


def avg_exceedance_loss(y: np.ndarray, var: np.ndarray) -> float:
    hits = y < var
    if hits.sum() == 0:
        return 0.0
    return float(np.mean(var[hits] - y[hits]))


def christoffersen_uc_test(y: np.ndarray, var: np.ndarray, alpha: float) -> dict:
    """
    Christoffersen (1998) unconditional coverage test.

    H0: The empirical violation rate equals alpha.

    Returns dict with test_stat, p_value, violations, n.
    """
    hits = (y < var).astype(int)
    n = len(hits)
    n1 = int(hits.sum())
    n0 = n - n1

    if n1 == 0 or n1 == n:
        return {
            "test_stat": np.nan,
            "p_value": np.nan,
            "violations": n1,
            "n": n,
            "empirical_rate": n1 / n,
        }

    pi_hat = n1 / n

    # Log-likelihood under H0 (rate = alpha)
    ll_0 = n1 * np.log(alpha) + n0 * np.log(1 - alpha)
    # Log-likelihood under H1 (rate = pi_hat)
    ll_1 = n1 * np.log(pi_hat) + n0 * np.log(1 - pi_hat)

    lr_stat = 2 * (ll_1 - ll_0)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "test_stat": float(lr_stat),
        "p_value": float(p_value),
        "violations": n1,
        "n": n,
        "empirical_rate": float(pi_hat),
    }


def christoffersen_cc_test(y: np.ndarray, var: np.ndarray, alpha: float) -> dict:
    """
    Christoffersen (1998) conditional coverage test (independence + UC).

    Returns dict with uc_stat, ind_stat, cc_stat, p_value_cc.
    """
    hits = (y < var).astype(int)
    n = len(hits)

    # UC part
    uc = christoffersen_uc_test(y, var, alpha)

    # Independence part: count transitions
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = hits[i - 1], hits[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Avoid log(0)
    eps = 1e-10

    # Under H0 (independence): pi = (n01 + n11) / (n - 1)
    n0x = n00 + n01
    n1x = n10 + n11
    pi_hat = (n01 + n11) / max(n - 1, 1)

    if n0x == 0 or n1x == 0 or pi_hat < eps or pi_hat > 1 - eps:
        return {
            "uc_stat": uc["test_stat"],
            "uc_p_value": uc["p_value"],
            "ind_stat": np.nan,
            "ind_p_value": np.nan,
            "cc_stat": np.nan,
            "cc_p_value": np.nan,
        }

    ll_0_ind = n0x * (
        (n00 / max(n0x, 1)) * np.log(1 - pi_hat + eps)
        + (n01 / max(n0x, 1)) * np.log(pi_hat + eps)
    ) + n1x * (
        (n10 / max(n1x, 1)) * np.log(1 - pi_hat + eps)
        + (n11 / max(n1x, 1)) * np.log(pi_hat + eps)
    )

    # Under H1: separate transition probs
    pi_01 = n01 / max(n0x, 1)
    pi_11 = n11 / max(n1x, 1)

    ll_1_ind = 0.0
    if n00 > 0:
        ll_1_ind += n00 * np.log(1 - pi_01 + eps)
    if n01 > 0:
        ll_1_ind += n01 * np.log(pi_01 + eps)
    if n10 > 0:
        ll_1_ind += n10 * np.log(1 - pi_11 + eps)
    if n11 > 0:
        ll_1_ind += n11 * np.log(pi_11 + eps)

    ind_stat = 2 * (ll_1_ind - ll_0_ind)
    ind_stat = max(ind_stat, 0.0)
    ind_p = 1 - stats.chi2.cdf(ind_stat, df=1)

    cc_stat = uc["test_stat"] + ind_stat if not np.isnan(uc["test_stat"]) else np.nan
    cc_p = 1 - stats.chi2.cdf(cc_stat, df=2) if not np.isnan(cc_stat) else np.nan

    return {
        "uc_stat": uc["test_stat"],
        "uc_p_value": uc["p_value"],
        "ind_stat": float(ind_stat),
        "ind_p_value": float(ind_p),
        "cc_stat": float(cc_stat) if not np.isnan(cc_stat) else np.nan,
        "cc_p_value": float(cc_p) if not np.isnan(cc_p) else np.nan,
    }


def basic_tail_metrics(y: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> dict:
    uc = christoffersen_uc_test(y, var, alpha)
    cc = christoffersen_cc_test(y, var, alpha)
    return {
        "coverage_rate": coverage_rate(y, var),
        "coverage_error": coverage_error(y, var, alpha),
        "avg_exceedance_loss": avg_exceedance_loss(y, var),
        "mean_var": float(np.mean(var)),
        "mean_es": float(np.mean(es)),
        "mean_y": float(np.mean(y)),
        "std_y": float(np.std(y)),
        "uc_stat": uc["test_stat"],
        "uc_p_value": uc["p_value"],
        "cc_stat": cc["cc_stat"],
        "cc_p_value": cc["cc_p_value"],
    }


def permutation_prediction_drift(preds: np.ndarray) -> dict[str, float]:
    centered = preds - preds[:, :1]
    abs_drift = np.abs(centered)
    return {
        "mean_abs_drift": float(abs_drift.mean()),
        "median_abs_drift": float(np.median(abs_drift)),
        "max_abs_drift": float(abs_drift.max()),
    }

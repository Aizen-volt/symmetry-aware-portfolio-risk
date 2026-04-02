#!/usr/bin/env python3
"""
Run Historical Simulation baselines for all dataset/portfolio/alpha combos.

This version aligns the HS benchmark with the forecasting task used by the
neural models:
    target at time t = sum_i w_i,t * r_i,t+1

Usage:
    python -m src.experiments.run_historical_simulation
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import load_ff_industry_returns
from src.data.portfolio import (
    compute_next_day_portfolio_returns,
    make_portfolio_weights,
)
from src.models.historical_simulation import (
    evaluate_historical_simulation,
    historical_simulation_var_es,
)
from src.training.metrics import basic_tail_metrics


DATASETS = ["industry12", "industry49"]
PORTFOLIOS = ["equal_weight", "vol_scaled"]
ALPHAS = [0.05, 0.01]
HS_WINDOW = 250


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset, portfolio, alpha in product(DATASETS, PORTFOLIOS, ALPHAS):
        print(f"\nHS: {dataset} / {portfolio} / alpha={alpha}")

        returns_df = load_ff_industry_returns(dataset, data_dir="data/raw")
        weights_df = make_portfolio_weights(
            returns_df=returns_df,
            portfolio=portfolio,
            vol_lookback=20,
        )

        # Align HS with the same one-step-ahead target definition as the neural models:
        # y_t = sum_i w_i,t * r_i,t+1
        port_ret = compute_next_day_portfolio_returns(
            returns_df=returns_df,
            weights_df=weights_df,
            horizon=1,
        )

        # Drop duplicate date indices if present
        port_ret = port_ret[~port_ret.index.duplicated(keep="first")].sort_index()

        result = evaluate_historical_simulation(
            portfolio_returns=port_ret,
            alpha=alpha,
            window=HS_WINDOW,
            train_frac=0.70,
            val_frac=0.15,
        )

        # Full HS predictions for the same aligned one-step-ahead target series
        hs = historical_simulation_var_es(
            port_ret,
            alpha=alpha,
            window=HS_WINDOW,
            min_periods=60,
        )

        aligned = pd.DataFrame({
            "y": port_ret,
            "var": hs["var"],
            "es": hs["es"],
        }).dropna()

        n = len(aligned)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        test_data = aligned.iloc[n_train + n_val:]

        if len(test_data) > 0:
            tail_metrics = basic_tail_metrics(
                y=test_data["y"].values,
                var=test_data["var"].values,
                es=test_data["es"].values,
                alpha=alpha,
            )
            result.update(tail_metrics)

        result["dataset"] = dataset
        result["portfolio"] = portfolio
        result["alpha"] = alpha
        result["model"] = "historical_sim"
        result["window"] = HS_WINDOW

        all_results.append(result)

        run_name = f"{dataset}__{portfolio}__historical_sim__alpha_{alpha:.2f}"
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        if len(test_data) > 0:
            test_data.to_csv(run_dir / "test_predictions.csv", index=True)

        cov_err = result.get("coverage_error", np.nan)
        avg_exc = result.get("avg_exceedance_loss", np.nan)
        print(f"  coverage_error={cov_err:.6f}  avg_exc={avg_exc:.6f}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(out_dir / "historical_simulation_summary.csv", index=False)
    print(f"\nSaved summary to {out_dir / 'historical_simulation_summary.csv'}")


if __name__ == "__main__":
    main()
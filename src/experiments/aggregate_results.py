"""
Aggregate results from multi-seed experiments, compute mean +/- std,
and produce paper-ready LaTeX tables.

Usage:
    python -m src.experiments.aggregate_results
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARTIFACTS_DIR = Path("artifacts")


def safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_dir_name(name: str) -> dict[str, Any] | None:
    """
    Parse:
      dataset__portfolio__model__alpha_0.05__seed_42
      dataset__portfolio__historical_sim__alpha_0.05
    """
    parts = name.split("__")

    if len(parts) == 5:
        dataset, portfolio, model, alpha_part, seed_part = parts
        alpha = float(alpha_part.replace("alpha_", ""))
        seed = int(seed_part.replace("seed_", ""))
        return {"dataset": dataset, "portfolio": portfolio, "model": model, "alpha": alpha, "seed": seed}

    if len(parts) == 4:
        dataset, portfolio, model, alpha_part = parts
        alpha = float(alpha_part.replace("alpha_", ""))
        return {"dataset": dataset, "portfolio": portfolio, "model": model, "alpha": alpha, "seed": None}

    return None


def collect_all_results(artifacts_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(artifacts_dir.iterdir()):
        if not run_dir.is_dir() or "__" not in run_dir.name:
            continue

        parsed = parse_run_dir_name(run_dir.name)
        if parsed is None:
            continue

        metrics = safe_read_json(run_dir / "metrics.json")
        if metrics is None:
            continue

        row = {**parsed}
        for key in ["test_loss", "coverage_rate", "coverage_error", "avg_exceedance_loss",
                     "mean_var", "mean_es", "uc_stat", "uc_p_value", "cc_stat", "cc_p_value",
                     "n_train", "n_val", "n_test", "n_total", "date_start", "date_end"]:
            row[key] = metrics.get(key)

        perm = safe_read_json(run_dir / "permutation_test.json")
        if perm:
            row["var_perm_mean_drift"] = perm.get("var_drift", {}).get("mean_abs_drift")
            row["var_perm_max_drift"] = perm.get("var_drift", {}).get("max_abs_drift")
            row["es_perm_mean_drift"] = perm.get("es_drift", {}).get("mean_abs_drift")
            row["es_perm_max_drift"] = perm.get("es_drift", {}).get("max_abs_drift")
            row["n_perm"] = perm.get("n_perm")

        row["run_dir"] = str(run_dir)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_multi_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (dataset, portfolio, model, alpha), compute mean ± std across seeds."""
    group_cols = ["dataset", "portfolio", "model", "alpha"]
    metric_cols = ["test_loss", "coverage_error", "avg_exceedance_loss",
                   "var_perm_mean_drift", "var_perm_max_drift",
                   "es_perm_mean_drift", "es_perm_max_drift",
                   "uc_p_value", "cc_p_value"]

    existing_metrics = [c for c in metric_cols if c in df.columns]

    agg_dict = {}
    for col in existing_metrics:
        agg_dict[col] = ["mean", "std", "count"]

    # Also keep first values of non-aggregated columns
    for col in ["n_train", "n_val", "n_test", "n_total", "date_start", "date_end"]:
        if col in df.columns:
            agg_dict[col] = "first"

    grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)
    grouped.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns]

    return grouped


def format_mean_std(mean_val, std_val, fmt=".4f") -> str:
    if pd.isna(mean_val):
        return "—"
    if pd.isna(std_val) or std_val == 0:
        return f"{mean_val:{fmt}}"
    return f"{mean_val:{fmt}} ± {std_val:{fmt}}"


def format_sci_mean_std(mean_val, std_val) -> str:
    if pd.isna(mean_val):
        return "—"
    if mean_val < 1e-6:
        return f"{mean_val:.2e}"
    if pd.isna(std_val) or std_val == 0:
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f} ± {std_val:.4f}"


def build_paper_forecasting_table(summary: pd.DataFrame) -> str:
    """Build consolidated LaTeX forecasting table."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\caption{Forecasting results across all configurations (mean $\pm$ std over 5 seeds). Lower is better for all metrics. HS = Historical Simulation baseline.}\label{tab:forecast_all}")
    lines.append(r"\small")
    lines.append(r"\begin{tabularx}{\textwidth}{llllCCCC}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Portfolio & $\alpha$ & Model & Test Loss & Coverage Error & Avg.\ Exc.\ Loss & UC $p$ \\")
    lines.append(r"\midrule")

    for _, row in summary.iterrows():
        ds = row["dataset"].replace("industry", "Ind")
        pf = "EW" if row["portfolio"] == "equal_weight" else "VS"
        alpha = f"{row['alpha']:.2f}"
        model = row["model"].replace("flatten_mlp", "Flatten-MLP").replace("set_attention", "Set-Attention").replace("historical_sim", "Hist.~Sim.")

        tl = format_mean_std(row.get("test_loss_mean"), row.get("test_loss_std"), ".6f")
        ce = format_mean_std(row.get("coverage_error_mean"), row.get("coverage_error_std"), ".4f")
        ae = format_mean_std(row.get("avg_exceedance_loss_mean"), row.get("avg_exceedance_loss_std"), ".4f")
        uc = format_mean_std(row.get("uc_p_value_mean"), row.get("uc_p_value_std"), ".3f")

        lines.append(f"{ds} & {pf} & {alpha} & {model} & {tl} & {ce} & {ae} & {uc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_paper_permutation_table(summary: pd.DataFrame) -> str:
    """Build consolidated LaTeX permutation table."""
    # Filter to neural models only (HS has no permutation test)
    neural = summary[summary["model"] != "historical_sim"].copy()

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\caption{Permutation robustness across all configurations (mean over 5 seeds, 100 permutations each). Lower is better.}\label{tab:perm_all}")
    lines.append(r"\small")
    lines.append(r"\begin{tabularx}{\textwidth}{llllCCCC}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Portfolio & $\alpha$ & Model & Mean VaR Drift & Max VaR Drift & Mean ES Drift & Max ES Drift \\")
    lines.append(r"\midrule")

    for _, row in neural.iterrows():
        ds = row["dataset"].replace("industry", "Ind")
        pf = "EW" if row["portfolio"] == "equal_weight" else "VS"
        alpha = f"{row['alpha']:.2f}"
        model = row["model"].replace("flatten_mlp", "Flatten-MLP").replace("set_attention", "Set-Attention")

        vmd = format_sci_mean_std(row.get("var_perm_mean_drift_mean"), row.get("var_perm_mean_drift_std"))
        vmx = format_sci_mean_std(row.get("var_perm_max_drift_mean"), row.get("var_perm_max_drift_std"))
        emd = format_sci_mean_std(row.get("es_perm_mean_drift_mean"), row.get("es_perm_mean_drift_std"))
        emx = format_sci_mean_std(row.get("es_perm_max_drift_mean"), row.get("es_perm_max_drift_std"))

        lines.append(f"{ds} & {pf} & {alpha} & {model} & {vmd} & {vmx} & {emd} & {emx} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    df = collect_all_results(ARTIFACTS_DIR)
    if df.empty:
        print("No results found.")
        return

    df.to_csv(ARTIFACTS_DIR / "all_runs.csv", index=False)

    summary = compute_multi_seed_summary(df)
    summary.to_csv(ARTIFACTS_DIR / "multi_seed_summary.csv", index=False)

    # Generate LaTeX tables
    forecast_tex = build_paper_forecasting_table(summary)
    perm_tex = build_paper_permutation_table(summary)

    (ARTIFACTS_DIR / "table_forecasting.tex").write_text(forecast_tex)
    (ARTIFACTS_DIR / "table_permutation.tex").write_text(perm_tex)

    print(f"Total runs: {len(df)}")
    print(f"Unique configs: {len(summary)}")
    print(f"Saved: all_runs.csv, multi_seed_summary.csv, table_forecasting.tex, table_permutation.tex")

    # Print summary
    print("\nForecasting summary:")
    cols = ["dataset", "portfolio", "alpha", "model"]
    for c in ["test_loss_mean", "test_loss_std", "coverage_error_mean", "avg_exceedance_loss_mean"]:
        if c in summary.columns:
            cols.append(c)
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()

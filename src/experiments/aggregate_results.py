from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ARTIFACTS_DIR = Path("artifacts")
OUTPUT_CSV = ARTIFACTS_DIR / "results_summary.csv"
OUTPUT_TEX = ARTIFACTS_DIR / "results_summary.tex"
OUTPUT_MD = ARTIFACTS_DIR / "results_summary.md"


def safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_dir_name(run_dir_name: str) -> dict[str, Any]:
    """
    Expected format:
      dataset__portfolio__model__alpha_0.05
    """
    parts = run_dir_name.split("__")
    if len(parts) != 4:
        raise ValueError(f"Unexpected run directory format: {run_dir_name}")

    dataset, portfolio, model, alpha_part = parts
    if not alpha_part.startswith("alpha_"):
        raise ValueError(f"Unexpected alpha format in: {run_dir_name}")

    alpha = float(alpha_part.replace("alpha_", ""))

    return {
        "dataset": dataset,
        "portfolio": portfolio,
        "model": model,
        "alpha": alpha,
    }


def extract_metrics_row(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    perm_path = run_dir / "permutation_test.json"

    metrics = safe_read_json(metrics_path)
    if metrics is None:
        return None

    row = parse_run_dir_name(run_dir.name)

    row["run_dir"] = str(run_dir)
    row["has_metrics"] = True
    row["has_permutation_test"] = perm_path.exists()

    # Main experiment metrics
    row["test_loss"] = metrics.get("test_loss")
    row["coverage_rate"] = metrics.get("coverage_rate")
    row["coverage_error"] = metrics.get("coverage_error")
    row["avg_exceedance_loss"] = metrics.get("avg_exceedance_loss")
    row["mean_var"] = metrics.get("mean_var")
    row["mean_es"] = metrics.get("mean_es")
    row["mean_y"] = metrics.get("mean_y")
    row["std_y"] = metrics.get("std_y")

    # Optional config echo
    config = metrics.get("config", {})
    row["lookback"] = config.get("lookback")
    row["vol_lookback"] = config.get("vol_lookback")
    row["min_assets"] = config.get("min_assets")
    row["batch_size"] = config.get("batch_size")
    row["seed"] = config.get("seed")

    # Permutation stats if available
    perm = safe_read_json(perm_path)
    if perm is not None:
        var_drift = perm.get("var_drift", {})
        es_drift = perm.get("es_drift", {})

        row["var_perm_mean_drift"] = var_drift.get("mean_abs_drift")
        row["var_perm_median_drift"] = var_drift.get("median_abs_drift")
        row["var_perm_max_drift"] = var_drift.get("max_abs_drift")

        row["es_perm_mean_drift"] = es_drift.get("mean_abs_drift")
        row["es_perm_median_drift"] = es_drift.get("median_abs_drift")
        row["es_perm_max_drift"] = es_drift.get("max_abs_drift")

        row["n_perm"] = perm.get("n_perm")
    else:
        row["var_perm_mean_drift"] = None
        row["var_perm_median_drift"] = None
        row["var_perm_max_drift"] = None
        row["es_perm_mean_drift"] = None
        row["es_perm_median_drift"] = None
        row["es_perm_max_drift"] = None
        row["n_perm"] = None

    return row


def collect_results(artifacts_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir.resolve()}")

    for run_dir in sorted(artifacts_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if "__" not in run_dir.name:
            continue

        try:
            row = extract_metrics_row(run_dir)
        except Exception as e:
            print(f"Skipping {run_dir.name}: {e}")
            continue

        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid run results found in: {artifacts_dir.resolve()}")

    df = pd.DataFrame(rows)

    sort_cols = ["alpha", "dataset", "portfolio", "model"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    preferred_cols = [
        "dataset",
        "portfolio",
        "alpha",
        "model",
        "test_loss",
        "coverage_rate",
        "coverage_error",
        "avg_exceedance_loss",
        "mean_var",
        "mean_es",
        "var_perm_mean_drift",
        "var_perm_median_drift",
        "var_perm_max_drift",
        "es_perm_mean_drift",
        "es_perm_median_drift",
        "es_perm_max_drift",
        "n_perm",
        "lookback",
        "vol_lookback",
        "min_assets",
        "batch_size",
        "seed",
        "has_metrics",
        "has_permutation_test",
        "run_dir",
    ]

    existing_cols = [c for c in preferred_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]

    return df


def build_forecasting_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "portfolio",
        "alpha",
        "model",
        "test_loss",
        "coverage_error",
        "avg_exceedance_loss",
        "coverage_rate",
        "mean_var",
        "mean_es",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    return out.sort_values(["alpha", "dataset", "portfolio", "model"]).reset_index(drop=True)


def build_permutation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "portfolio",
        "alpha",
        "model",
        "var_perm_mean_drift",
        "var_perm_median_drift",
        "var_perm_max_drift",
        "es_perm_mean_drift",
        "es_perm_median_drift",
        "es_perm_max_drift",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    return out.sort_values(["alpha", "dataset", "portfolio", "model"]).reset_index(drop=True)


def format_numeric_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: (
                    ""
                    if pd.isna(x)
                    else f"{x:.6g}" if abs(float(x)) < 1e-3 or abs(float(x)) >= 1e3
                    else f"{x:.6f}"
                )
            )
    return out


def dataframe_to_markdown(df: pd.DataFrame, title: str) -> str:
    disp = format_numeric_for_display(df)
    lines = [f"# {title}", "", disp.to_markdown(index=False), ""]
    return "\n".join(lines)


def save_latex_table(df: pd.DataFrame, path: Path) -> None:
    disp = format_numeric_for_display(df)
    latex = disp.to_latex(index=False, escape=False)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    df = collect_results(ARTIFACTS_DIR)

    # Full summary
    df.to_csv(OUTPUT_CSV, index=False)

    # Main paper-facing tables
    forecasting_df = build_forecasting_table(df)
    permutation_df = build_permutation_table(df)

    # Save one combined markdown report
    md_parts = [
        dataframe_to_markdown(df, "Full Results Summary"),
        dataframe_to_markdown(forecasting_df, "Forecasting Metrics"),
        dataframe_to_markdown(permutation_df, "Permutation Robustness Metrics"),
    ]
    OUTPUT_MD.write_text("\n".join(md_parts), encoding="utf-8")

    # Save LaTeX from full summary by default
    save_latex_table(df, OUTPUT_TEX)

    print(f"Saved CSV : {OUTPUT_CSV.resolve()}")
    print(f"Saved TEX : {OUTPUT_TEX.resolve()}")
    print(f"Saved MD  : {OUTPUT_MD.resolve()}")

    print("\nForecasting table preview:")
    print(format_numeric_for_display(forecasting_df).to_string(index=False))

    print("\nPermutation table preview:")
    print(format_numeric_for_display(permutation_df).to_string(index=False))


if __name__ == "__main__":
    main()
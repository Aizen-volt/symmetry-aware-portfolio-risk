"""
Generate all paper figures from experiment results.

Produces:
  - fig_forecasting_metrics.pdf   (2-panel vertical: coverage error + exceedance loss)
  - fig_permutation_drift.pdf     (2-panel vertical: VaR drift + ES drift)
  - fig_forecast_timeseries.pdf   (3-panel vertical: one panel per model)

Usage:
    python -m src.experiments.plot_results
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ARTIFACTS_DIR = Path("artifacts")
OUT_DIR = Path("paper/figures")


def load_summary() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "multi_seed_summary.csv"
    if not path.exists():
        path = ARTIFACTS_DIR / "results_summary.csv"
    return pd.read_csv(path)


def _setting_labels(settings_order: list[tuple[str, str, float]]) -> list[str]:
    labels = []
    for ds, pf, a in settings_order:
        ds_short = ds.replace("industry", "Ind")
        pf_short = "EW" if pf == "equal_weight" else "VS"
        labels.append(f"{ds_short}-{pf_short}\nα={a:.0%}")
    return labels


# ── Figure 1: Consolidated forecasting metrics ─────────────────────────

def plot_forecasting_metrics(df: pd.DataFrame) -> None:
    """Bar chart: coverage error + avg exceedance loss stacked vertically."""
    ce_col = "coverage_error_mean" if "coverage_error_mean" in df.columns else "coverage_error"
    ae_col = "avg_exceedance_loss_mean" if "avg_exceedance_loss_mean" in df.columns else "avg_exceedance_loss"

    models = ["historical_sim", "flatten_mlp", "set_attention"]
    model_labels = {
        "flatten_mlp": "Flatten-MLP",
        "set_attention": "Set-Attention",
        "historical_sim": "Hist. Sim.",
    }
    colors = {
        "flatten_mlp": "#1f77b4",
        "set_attention": "#ff7f0e",
        "historical_sim": "#2ca02c",
    }

    settings_order = [
        ("industry12", "equal_weight", 0.05),
        ("industry12", "vol_scaled", 0.05),
        ("industry49", "equal_weight", 0.05),
        ("industry49", "vol_scaled", 0.05),
        ("industry12", "equal_weight", 0.01),
        ("industry12", "vol_scaled", 0.01),
        ("industry49", "equal_weight", 0.01),
        ("industry49", "vol_scaled", 0.01),
    ]

    settings_labels = _setting_labels(settings_order)

    n_settings = len(settings_order)
    n_models = len(models)
    width = 0.78 / n_models
    x = np.arange(n_settings)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    panels = [
        (ce_col, "Coverage error", "Coverage Error Across Configurations"),
        (ae_col, "Average exceedance loss", "Average Exceedance Loss Across Configurations"),
    ]

    for panel_idx, (metric_col, ylabel, title) in enumerate(panels):
        ax = axes[panel_idx]

        for m_idx, model in enumerate(models):
            vals = []
            for ds, pf, a in settings_order:
                mask = (
                    (df["dataset"] == ds)
                    & (df["portfolio"] == pf)
                    & (df["alpha"] == a)
                    & (df["model"] == model)
                )
                sub = df[mask]
                vals.append(float(sub[metric_col].iloc[0]) if len(sub) > 0 else np.nan)

            offset = (m_idx - n_models / 2 + 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=model_labels.get(model, model),
                color=colors.get(model, "gray"),
                edgecolor="black",
                linewidth=0.4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(settings_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(axis="y", alpha=0.30)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)

        if panel_idx == 0:
            ax.legend(frameon=False, fontsize=10, ncol=3)

    out = OUT_DIR / "fig_forecasting_metrics.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)

    # backward-compatible extra copy
    out2 = OUT_DIR / "fig_coverage_error.pdf"
    fig.savefig(out2, bbox_inches="tight", dpi=300)

    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: Consolidated permutation drift ───────────────────────────

def plot_permutation_drift(df: pd.DataFrame) -> None:
    """Log-scale bar chart: VaR drift + ES drift stacked vertically."""
    var_col = "var_perm_mean_drift_mean" if "var_perm_mean_drift_mean" in df.columns else "var_perm_mean_drift"
    es_col = "es_perm_mean_drift_mean" if "es_perm_mean_drift_mean" in df.columns else "es_perm_mean_drift"

    neural = df[df["model"] != "historical_sim"].copy()
    models = ["flatten_mlp", "set_attention"]
    model_labels = {"flatten_mlp": "Flatten-MLP", "set_attention": "Set-Attention"}
    colors = {"flatten_mlp": "#1f77b4", "set_attention": "#ff7f0e"}

    settings_order = [
        ("industry12", "equal_weight", 0.05),
        ("industry12", "vol_scaled", 0.05),
        ("industry49", "equal_weight", 0.05),
        ("industry49", "vol_scaled", 0.05),
        ("industry12", "equal_weight", 0.01),
        ("industry12", "vol_scaled", 0.01),
        ("industry49", "equal_weight", 0.01),
        ("industry49", "vol_scaled", 0.01),
    ]
    settings_labels = _setting_labels(settings_order)

    n_settings = len(settings_order)
    width = 0.34
    x = np.arange(n_settings)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    panels = [
        (var_col, "Mean absolute VaR drift", "VaR Permutation Sensitivity"),
        (es_col, "Mean absolute ES drift", "ES Permutation Sensitivity"),
    ]

    for panel_idx, (metric_col, ylabel, title) in enumerate(panels):
        ax = axes[panel_idx]

        for m_idx, model in enumerate(models):
            vals = []
            for ds, pf, a in settings_order:
                mask = (
                    (neural["dataset"] == ds)
                    & (neural["portfolio"] == pf)
                    & (neural["alpha"] == a)
                    & (neural["model"] == model)
                )
                sub = neural[mask]
                if len(sub) > 0 and not pd.isna(sub[metric_col].iloc[0]):
                    vals.append(max(float(sub[metric_col].iloc[0]), 1e-12))
                else:
                    vals.append(1e-12)

            offset = (m_idx - 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=model_labels[model],
                color=colors[model],
                edgecolor="black",
                linewidth=0.4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(settings_labels, fontsize=10)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(axis="y", alpha=0.30, which="both")
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)

        if panel_idx == 0:
            ax.legend(frameon=False, fontsize=10, ncol=2)

    out = OUT_DIR / "fig_permutation_drift.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)

    # backward-compatible extra copies
    out2 = OUT_DIR / "fig_var_drift.pdf"
    fig.savefig(out2, bbox_inches="tight", dpi=300)
    out3 = OUT_DIR / "fig_es_drift.pdf"
    fig.savefig(out3, bbox_inches="tight", dpi=300)

    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Model-by-model forecast panels ───────────────────────────

def _resolve_prediction_dir(dataset: str, portfolio: str, model_key: str, alpha: float) -> Path | None:
    if model_key == "historical_sim":
        candidates = [
            ARTIFACTS_DIR / f"{dataset}__{portfolio}__historical_sim__alpha_{alpha:.2f}" / "test_predictions.csv",
        ]
    else:
        candidates = [
            ARTIFACTS_DIR / f"{dataset}__{portfolio}__{model_key}__alpha_{alpha:.2f}__seed_42" / "test_predictions.csv",
            ARTIFACTS_DIR / f"{dataset}__{portfolio}__{model_key}__alpha_{alpha:.2f}" / "test_predictions.csv",
        ]

    for path in candidates:
        if path.exists():
            return path
    return None


def plot_forecast_timeseries(
    dataset: str = "industry12",
    portfolio: str = "equal_weight",
    alpha: float = 0.05,
    last_n_days: int = 300,
    output_name: str | None = None,
) -> None:
    """
    Plot a clean 3-panel vertical comparison:
      1) Hist. Sim.
      2) Flatten-MLP
      3) Set-Attention
    """
    model_order = [
        ("historical_sim", "Hist. Sim."),
        ("flatten_mlp", "Flatten-MLP"),
        ("set_attention", "Set-Attention"),
    ]
    colors = {
        "historical_sim": "#2ca02c",
        "flatten_mlp": "#1f77b4",
        "set_attention": "#ff7f0e",
    }

    loaded = []
    for model_key, model_label in model_order:
        path = _resolve_prediction_dir(dataset, portfolio, model_key, alpha)
        if path is None:
            print(f"  Skipping {model_label}: predictions not found")
            continue
        preds = pd.read_csv(path).tail(last_n_days).reset_index(drop=True)
        loaded.append((model_key, model_label, preds))

    if not loaded:
        print("  No prediction files found; skipping forecast figure.")
        return

    # common y-limits across all panels
    y_min = min(float(preds[["y", "var", "es"]].min().min()) for _, _, preds in loaded if {"y", "var", "es"}.issubset(preds.columns))
    y_max = max(float(preds[["y", "var", "es"]].max().max()) for _, _, preds in loaded if {"y", "var", "es"}.issubset(preds.columns))
    pad = 0.05 * max(1e-6, y_max - y_min)
    y_limits = (y_min - pad, y_max + pad)

    fig, axes = plt.subplots(len(loaded), 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    if len(loaded) == 1:
        axes = [axes]

    for ax, (model_key, model_label, preds) in zip(axes, loaded):
        t = np.arange(len(preds))
        color = colors.get(model_key, "gray")

        ax.plot(t, preds["y"], color="black", alpha=0.55, linewidth=0.8, label="Realized return")
        ax.plot(t, preds["var"], color=color, linewidth=1.4, label="VaR")
        if "es" in preds.columns:
            ax.plot(t, preds["es"], color=color, linewidth=1.2, linestyle="--", alpha=0.9, label="ES")

        exceed = preds["y"] < preds["var"]
        ax.fill_between(
            t,
            preds["y"],
            preds["var"],
            where=exceed,
            color="red",
            alpha=0.18,
            interpolate=True,
            label="VaR exceedance",
        )

        ax.set_ylabel("Return", fontsize=11)
        ax.set_title(model_label, fontsize=12, pad=8)
        ax.set_ylim(*y_limits)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.tick_params(axis="y", labelsize=10)

        # compact legend inside each panel
        ax.legend(loc="lower left", fontsize=8, frameon=True, ncol=4)

    axes[-1].set_xlabel("Test observation index", fontsize=11)

    ds_label = dataset.replace("industry", "Industry ")
    pf_label = "Equal-Weight" if portfolio == "equal_weight" else "Vol-Scaled"
    fig.suptitle(f"VaR/ES Forecast Comparison — {ds_label}, {pf_label}, α={alpha}", fontsize=14)

    if output_name is None:
        output_name = "fig_forecast_timeseries.pdf"

    out = OUT_DIR / output_name
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_summary()
    plot_forecasting_metrics(df)
    plot_permutation_drift(df)

    # Main-paper representative figure
    plot_forecast_timeseries(
        dataset="industry12",
        portfolio="equal_weight",
        alpha=0.05,
        last_n_days=300,
        output_name="fig_forecast_timeseries.pdf",
    )

    print("\nAll figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
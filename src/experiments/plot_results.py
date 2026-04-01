from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ARTIFACTS_CSV = Path(__file__).resolve().parents[1] / "14593705-9da6-4710-ac41-5df115534d32.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "revised_paper"

def prepare_df():
    df = pd.read_csv(ARTIFACTS_CSV)
    df["dataset_disp"] = df["dataset"].map({"industry12": "Ind12", "industry49": "Ind49"})
    df["portfolio_disp"] = df["portfolio"].map({"equal_weight": "EW", "vol_scaled": "VS"})
    df["alpha_disp"] = df["alpha"].map(lambda x: "0.05" if abs(x - 0.05) < 1e-12 else "0.01")
    df["model_disp"] = df["model"].map({"flatten_mlp": "Flatten-MLP", "set_attention": "Set-Attention"})
    df["setting"] = df["dataset_disp"] + "-" + df["portfolio_disp"] + "\n" + r"$\alpha$=" + df["alpha_disp"]
    order = [
        ("industry12", "equal_weight", 0.05),
        ("industry12", "vol_scaled", 0.05),
        ("industry49", "equal_weight", 0.05),
        ("industry49", "vol_scaled", 0.05),
        ("industry12", "equal_weight", 0.01),
        ("industry12", "vol_scaled", 0.01),
        ("industry49", "equal_weight", 0.01),
        ("industry49", "vol_scaled", 0.01),
    ]
    order_map = {k: i for i, k in enumerate(order)}
    df["ord"] = df.apply(lambda r: order_map[(r["dataset"], r["portfolio"], r["alpha"])], axis=1)
    return df.sort_values(["ord", "model_disp"]).reset_index(drop=True)

def grouped_values(df, metric):
    settings = df["setting"].drop_duplicates().tolist()
    flat = []
    setm = []
    for s in settings:
        sub = df[df["setting"] == s]
        flat.append(float(sub.loc[sub["model"] == "flatten_mlp", metric].iloc[0]))
        setm.append(float(sub.loc[sub["model"] == "set_attention", metric].iloc[0]))
    return settings, np.array(flat), np.array(setm)

def save_forecasting_plot(df):
    settings, flat_cov, set_cov = grouped_values(df, "coverage_error")
    _, flat_exc, set_exc = grouped_values(df, "avg_exceedance_loss")

    x = np.arange(len(settings))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)

    axes[0].bar(x - width/2, flat_cov, width, label="Flatten-MLP")
    axes[0].bar(x + width/2, set_cov, width, label="Set-Attention")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(settings)
    axes[0].set_ylabel("Coverage error")
    axes[0].set_title("Calibration error")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width/2, flat_exc, width, label="Flatten-MLP")
    axes[1].bar(x + width/2, set_exc, width, label="Set-Attention")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(settings)
    axes[1].set_ylabel("Average exceedance loss")
    axes[1].set_title("Tail severity")

    out = OUT_DIR / "fig_forecasting_metrics.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

def save_permutation_plot(df):
    settings, flat_var, set_var = grouped_values(df, "var_perm_mean_drift")
    _, flat_es, set_es = grouped_values(df, "es_perm_mean_drift")

    x = np.arange(len(settings))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)

    axes[0].bar(x - width/2, flat_var, width, label="Flatten-MLP")
    axes[0].bar(x + width/2, set_var, width, label="Set-Attention")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(settings)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean absolute VaR drift")
    axes[0].set_title("Permutation sensitivity: VaR")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width/2, flat_es, width, label="Flatten-MLP")
    axes[1].bar(x + width/2, set_es, width, label="Set-Attention")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(settings)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean absolute ES drift")
    axes[1].set_title("Permutation sensitivity: ES")

    out = OUT_DIR / "fig_permutation_drift.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_df()
    save_forecasting_plot(df)
    save_permutation_plot(df)
    print("Saved:", OUT_DIR / "fig_forecasting_metrics.pdf")
    print("Saved:", OUT_DIR / "fig_permutation_drift.pdf")

if __name__ == "__main__":
    main()

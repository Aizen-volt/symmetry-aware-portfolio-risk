"""
Run permutation robustness test on a trained model checkpoint.

Updated: default n_perm=100 (was 20).

Usage:
    python -m src.experiments.run_permutation_test \\
        --dataset industry12 --portfolio equal_weight --model set_attention \\
        --checkpoint artifacts/.../model.pt --n_perm 100
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.loaders import infer_market_series, load_ff_industry_returns
from src.data.panel_dataset import MultiAssetPanelDataset
from src.data.portfolio import make_portfolio_weights
from src.models.flatten_temporal_mlp import FlattenTemporalMLPVaRES
from src.models.set_attention_var_es import SetAttentionVaRESModel
from src.training.metrics import permutation_prediction_drift


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(model_name, n_assets, lookback, in_features):
    if model_name == "flatten_mlp":
        return FlattenTemporalMLPVaRES(
            n_assets=n_assets, lookback=lookback, in_features=in_features,
            hidden_dim=512, dropout=0.1,
        )
    if model_name == "set_attention":
        return SetAttentionVaRESModel(
            in_features=in_features, emb_dim=128, temporal_hidden_dim=64,
            num_attention_blocks=2, num_heads=4, ff_mult=4, dropout=0.1,
        )
    raise ValueError(model_name)


@torch.no_grad()
def collect_permutation_preds(model, loader, device, n_perm=100):
    model.eval()
    var_all, es_all = [], []

    for batch in loader:
        x = batch["x"].to(device)
        asset_mask = batch["asset_mask"].to(device)
        weights = batch["weights"].to(device)
        B, N, L, Fdim = x.shape
        batch_var, batch_es = [], []

        for p in range(n_perm):
            perm = torch.arange(N, device=device) if p == 0 else torch.randperm(N, device=device)
            out = model(x[:, perm], asset_mask[:, perm], weights[:, perm])
            batch_var.append(out["var"].cpu().numpy())
            batch_es.append(out["es"].cpu().numpy())

        var_all.append(np.stack(batch_var, axis=1))
        es_all.append(np.stack(batch_es, axis=1))

    return np.concatenate(var_all, axis=0), np.concatenate(es_all, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--portfolio", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--vol_lookback", type=int, default=20)
    parser.add_argument("--min_assets", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_perm", type=int, default=100)  # Updated from 20
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    returns_df = load_ff_industry_returns(args.dataset, data_dir="data/raw")
    market_series = infer_market_series(returns_df)
    weights_df = make_portfolio_weights(returns_df=returns_df, portfolio=args.portfolio, vol_lookback=args.vol_lookback)

    dataset = MultiAssetPanelDataset(
        returns_df=returns_df, market_series=market_series, weights_df=weights_df,
        lookback=args.lookback, horizon=1, min_assets=args.min_assets,
        vol_lookback=args.vol_lookback, normalize_inputs=True,
    )
    dataset.set_normalizer(ckpt["feature_mean"], ckpt["feature_std"])

    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    test_idx = list(range(n_train + n_val, n))

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size if device.type == "cuda" else min(args.batch_size, 64),
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    sample0 = dataset[0]
    model = make_model(args.model, sample0["x"].shape[0], args.lookback, sample0["x"].shape[-1]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    var_preds, es_preds = collect_permutation_preds(model, test_loader, device, args.n_perm)

    out_dir = Path(args.checkpoint).resolve().parent
    with open(out_dir / "permutation_test.json", "w") as f:
        json.dump({
            "var_drift": permutation_prediction_drift(var_preds),
            "es_drift": permutation_prediction_drift(es_preds),
            "n_perm": args.n_perm,
            "dataset": args.dataset,
            "portfolio": args.portfolio,
            "model": args.model,
        }, f, indent=2)

    print(f"Saved permutation test: {out_dir / 'permutation_test.json'}")


if __name__ == "__main__":
    main()

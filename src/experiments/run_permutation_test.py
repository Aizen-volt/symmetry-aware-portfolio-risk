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


def make_model(model_name: str, n_assets: int, lookback: int, in_features: int) -> torch.nn.Module:
    if model_name == "flatten_mlp":
        return FlattenTemporalMLPVaRES(
            n_assets=n_assets,
            lookback=lookback,
            in_features=in_features,
            hidden_dim=512,
            dropout=0.1,
        )
    if model_name == "set_attention":
        return SetAttentionVaRESModel(
            in_features=in_features,
            emb_dim=128,
            temporal_hidden_dim=64,
            num_attention_blocks=2,
            num_heads=4,
            ff_mult=4,
            dropout=0.1,
        )
    raise ValueError(model_name)


@torch.no_grad()
def collect_permutation_preds(model, loader, device, n_perm: int = 20):
    model.eval()

    var_preds_all = []
    es_preds_all = []

    for batch in loader:
        x = batch["x"].to(device)
        asset_mask = batch["asset_mask"].to(device)
        weights = batch["weights"].to(device)

        B, N, L, Fdim = x.shape
        batch_var = []
        batch_es = []

        for p in range(n_perm):
            if p == 0:
                perm = torch.arange(N, device=device)
            else:
                perm = torch.randperm(N, device=device)

            x_p = x[:, perm, :, :]
            mask_p = asset_mask[:, perm]
            weights_p = weights[:, perm]

            out = model(x_p, mask_p, weights_p)
            batch_var.append(out["var"].detach().cpu().numpy())
            batch_es.append(out["es"].detach().cpu().numpy())

        var_preds_all.append(np.stack(batch_var, axis=1))  # [B, P]
        es_preds_all.append(np.stack(batch_es, axis=1))    # [B, P]

    return np.concatenate(var_preds_all, axis=0), np.concatenate(es_preds_all, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["industry12", "industry49"])
    parser.add_argument("--portfolio", type=str, required=True, choices=["equal_weight", "vol_scaled"])
    parser.add_argument("--model", type=str, required=True, choices=["flatten_mlp", "set_attention"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--vol_lookback", type=int, default=20)
    parser.add_argument("--min_assets", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_perm", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    returns_df = load_ff_industry_returns(args.dataset, data_dir="data/raw")
    market_series = infer_market_series(returns_df)
    weights_df = make_portfolio_weights(
        returns_df=returns_df,
        portfolio=args.portfolio,
        vol_lookback=args.vol_lookback,
    )

    dataset = MultiAssetPanelDataset(
        returns_df=returns_df,
        market_series=market_series,
        weights_df=weights_df,
        lookback=args.lookback,
        horizon=1,
        min_assets=args.min_assets,
        vol_lookback=args.vol_lookback,
        normalize_inputs=True,
    )
    dataset.set_normalizer(ckpt["feature_mean"], ckpt["feature_std"])

    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    test_idx = list(range(n_train + n_val, n))

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size if device.type == "cuda" else min(args.batch_size, 64),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    sample0 = dataset[0]
    model = make_model(
        args.model,
        n_assets=sample0["x"].shape[0],
        lookback=args.lookback,
        in_features=sample0["x"].shape[-1],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    var_preds, es_preds = collect_permutation_preds(
        model=model,
        loader=test_loader,
        device=device,
        n_perm=args.n_perm,
    )

    var_stats = permutation_prediction_drift(var_preds)
    es_stats = permutation_prediction_drift(es_preds)

    out_dir = Path(args.checkpoint).resolve().parent
    with open(out_dir / "permutation_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "var_drift": var_stats,
                "es_drift": es_stats,
                "n_perm": args.n_perm,
                "dataset": args.dataset,
                "portfolio": args.portfolio,
                "model": args.model,
            },
            f,
            indent=2,
        )

    print("Permutation test results")
    print("VaR drift:", var_stats)
    print("ES drift :", es_stats)
    print(f"Saved to: {out_dir / 'permutation_test.json'}")


if __name__ == "__main__":
    main()
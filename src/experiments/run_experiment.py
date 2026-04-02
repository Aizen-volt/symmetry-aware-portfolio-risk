"""
Train a single model configuration. Updated to include seed in output directory name.

Usage:
    python -m src.experiments.run_experiment \\
        --dataset industry12 --portfolio equal_weight \\
        --model set_attention --alpha 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.loaders import infer_market_series, load_ff_industry_returns
from src.data.panel_dataset import MultiAssetPanelDataset
from src.data.portfolio import make_portfolio_weights
from src.models.flatten_temporal_mlp import FlattenTemporalMLPVaRES
from src.models.set_attention_var_es import SetAttentionVaRESModel
from src.training.engine import run_eval_epoch, run_train_epoch
from src.training.losses import StableVaRESLoss
from src.training.metrics import basic_tail_metrics


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(dataset, train_idx, val_idx, test_idx, batch_size, num_workers):
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, test_loader


def make_model(model_name: str, n_assets: int, lookback: int, in_features: int) -> torch.nn.Module:
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
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["industry12", "industry49"])
    parser.add_argument("--portfolio", type=str, required=True, choices=["equal_weight", "vol_scaled"])
    parser.add_argument("--model", type=str, required=True, choices=["flatten_mlp", "set_attention"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--vol_lookback", type=int, default=20)
    parser.add_argument("--min_assets", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    returns_df = load_ff_industry_returns(args.dataset, data_dir="data/raw")
    market_series = infer_market_series(returns_df)
    weights_df = make_portfolio_weights(
        returns_df=returns_df, portfolio=args.portfolio, vol_lookback=args.vol_lookback,
    )

    dataset = MultiAssetPanelDataset(
        returns_df=returns_df, market_series=market_series, weights_df=weights_df,
        lookback=args.lookback, horizon=1, min_assets=args.min_assets,
        vol_lookback=args.vol_lookback, normalize_inputs=False,
    )

    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n))

    dataset.fit_normalizer(indices=train_idx)
    dataset.normalize_inputs = True

    train_loader, val_loader, test_loader = build_loaders(
        dataset=dataset, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        batch_size=args.batch_size if device.type == "cuda" else min(args.batch_size, 64),
        num_workers=args.num_workers,
    )

    sample0 = dataset[0]
    n_assets = sample0["x"].shape[0]
    in_features = sample0["x"].shape[-1]

    model = make_model(args.model, n_assets, args.lookback, in_features).to(device)

    loss_fn = StableVaRESLoss(
        alpha=args.alpha, lambda_var=1.0, lambda_es=0.5,
        lambda_order=10.0, lambda_sign=1.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        train_res = run_train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
            device=device, scaler=scaler, grad_clip_norm=1.0, use_amp=True,
        )
        val_res, _ = run_eval_epoch(
            model=model, loader=val_loader, loss_fn=loss_fn, device=device, use_amp=True,
        )
        scheduler.step(val_res.loss)

        print(f"Epoch {epoch:03d} | train={train_res.loss:.6f} | val={val_res.loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_res.loss < best_val_loss:
            best_val_loss = val_res.loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No valid model state captured.")

    model.load_state_dict(best_state)

    test_res, test_payload = run_eval_epoch(
        model=model, loader=test_loader, loss_fn=loss_fn, device=device, use_amp=True,
    )
    metrics = basic_tail_metrics(
        y=test_payload["y"], var=test_payload["var"],
        es=test_payload["es"], alpha=args.alpha,
    )

    # Updated naming: include seed
    run_name = f"{args.dataset}__{args.portfolio}__{args.model}__alpha_{args.alpha:.2f}__seed_{args.seed}"
    out_dir = Path("artifacts") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "y": test_payload["y"], "var": test_payload["var"], "es": test_payload["es"],
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_loss": float(test_res.loss),
            **metrics,
            "config": vars(args),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": len(test_idx),
            "n_total": n,
            "date_start": str(returns_df.index.min().date()),
            "date_end": str(returns_df.index.max().date()),
        }, f, indent=2)

    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": dataset.feature_mean,
        "feature_std": dataset.feature_std,
        "config": vars(args),
        "n_assets": n_assets,
        "in_features": in_features,
    }, out_dir / "model.pt")

    print(f"\ntest_loss = {test_res.loss:.6f}")
    for k, v in metrics.items():
        print(f"{k:20s} = {v:.6f}" if isinstance(v, float) else f"{k:20s} = {v}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

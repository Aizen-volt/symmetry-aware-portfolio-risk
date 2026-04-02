from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

@dataclass
class EpochResult:
    loss: float
    n_obs: int

def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

def run_train_epoch(model, loader, optimizer, loss_fn, device, scaler=None, grad_clip_norm=1.0, use_amp=True):
    model.train()
    total_loss = 0.0; total_n = 0
    amp_enabled = use_amp and device.type == "cuda"
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            out = model(batch["x"], batch["asset_mask"], batch["weights"])
            loss = loss_fn(batch["y"], out["var"], out["es"])
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip_norm: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        bs = batch["y"].shape[0]; total_loss += float(loss.item()) * bs; total_n += bs
    return EpochResult(loss=total_loss / max(total_n, 1), n_obs=total_n)

@torch.no_grad()
def run_eval_epoch(model, loader, loss_fn, device, use_amp=True):
    model.eval()
    total_loss = 0.0; total_n = 0
    ys, vars_, ess = [], [], []
    amp_enabled = use_amp and device.type == "cuda"
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            out = model(batch["x"], batch["asset_mask"], batch["weights"])
            loss = loss_fn(batch["y"], out["var"], out["es"])
        bs = batch["y"].shape[0]; total_loss += float(loss.item()) * bs; total_n += bs
        ys.append(batch["y"].cpu().numpy()); vars_.append(out["var"].cpu().numpy()); ess.append(out["es"].cpu().numpy())
    payload = {"y": np.concatenate(ys), "var": np.concatenate(vars_), "es": np.concatenate(ess)}
    return EpochResult(loss=total_loss / max(total_n, 1), n_obs=total_n), payload

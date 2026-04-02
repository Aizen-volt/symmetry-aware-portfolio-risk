from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlattenTemporalMLPVaRES(nn.Module):
    def __init__(self, n_assets, lookback, in_features, hidden_dim=512, dropout=0.1):
        super().__init__()
        input_dim = n_assets * lookback * in_features + n_assets
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.var_out = nn.Linear(hidden_dim // 2, 1)
        self.gap_out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, asset_mask, weights=None):
        B, N, L, Fdim = x.shape
        x = x * asset_mask[:, :, None, None]
        flat_x = x.reshape(B, N * L * Fdim)
        if weights is None:
            weights = asset_mask / asset_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        z = torch.cat([flat_x, weights], dim=1)
        h = self.backbone(z)
        var_raw = self.var_out(h).squeeze(-1)
        gap_raw = self.gap_out(h).squeeze(-1)
        var = -0.1 * F.softplus(var_raw)
        gap = 0.1 * F.softplus(gap_raw)
        es = var - gap
        return {"var": var, "es": es, "embedding": h}

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mean(x, mask, dim):
    mask_exp = mask.unsqueeze(-1).to(x.dtype)
    num = (x * mask_exp).sum(dim=dim)
    den = mask_exp.sum(dim=dim).clamp_min(1e-6)
    return num / den

class TemporalConvEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim=64, emb_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, emb_dim, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm1d(emb_dim),
        )
        self.proj = nn.Linear(emb_dim, emb_dim)
    def forward(self, x):
        B, N, L, Fdim = x.shape
        x = x.reshape(B * N, L, Fdim).transpose(1, 2)
        h = self.net(x)
        h = h.mean(dim=-1)
        h = self.proj(h)
        return h.reshape(B, N, -1)

class SetAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * dim, dim), nn.Dropout(dropout))
    def forward(self, x, asset_mask):
        key_padding_mask = ~(asset_mask.bool())
        z = self.ln1(x)
        attn_out, _ = self.attn(z, z, z, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class WeightedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, 1))
    def forward(self, h, asset_mask, weights=None):
        if weights is None:
            return masked_mean(h, asset_mask, dim=1)
        w = weights.unsqueeze(-1)
        logits = self.gate(torch.cat([h, w], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~asset_mask.bool(), float("-inf"))
        attn = torch.softmax(logits, dim=1)
        abs_w = weights.abs() * asset_mask
        abs_w = abs_w / abs_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mix = 0.5 * attn + 0.5 * abs_w
        mix = mix / mix.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(h * mix.unsqueeze(-1), dim=1)

class VaRESHead(nn.Module):
    def __init__(self, dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.var_out = nn.Linear(hidden_dim, 1)
        self.gap_out = nn.Linear(hidden_dim, 1)
    def forward(self, z):
        h = self.backbone(z)
        var = -0.1 * F.softplus(self.var_out(h).squeeze(-1))
        gap = 0.1 * F.softplus(self.gap_out(h).squeeze(-1))
        return var, var - gap

class SetAttentionVaRESModel(nn.Module):
    def __init__(self, in_features, emb_dim=128, temporal_hidden_dim=64, num_attention_blocks=2, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.encoder = TemporalConvEncoder(in_features, temporal_hidden_dim, emb_dim, dropout)
        self.blocks = nn.ModuleList([SetAttentionBlock(emb_dim, num_heads, ff_mult, dropout) for _ in range(num_attention_blocks)])
        self.pool = WeightedPooling(emb_dim)
        self.head = VaRESHead(emb_dim, emb_dim, dropout)
    def forward(self, x, asset_mask, weights=None):
        h = self.encoder(x)
        h = h * asset_mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, asset_mask)
            h = h * asset_mask.unsqueeze(-1)
        pooled = self.pool(h, asset_mask, weights)
        var, es = self.head(pooled)
        return {"var": var, "es": es, "embedding": pooled}

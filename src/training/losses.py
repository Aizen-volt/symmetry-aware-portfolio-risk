from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = float(alpha)
    def forward(self, y, q):
        u = y - q
        return torch.maximum(self.alpha * u, (self.alpha - 1.0) * u).mean()

class StableVaRESLoss(nn.Module):
    def __init__(self, alpha, lambda_var=1.0, lambda_es=1.0, lambda_order=10.0, lambda_sign=1.0, eps=1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.lambda_var = float(lambda_var)
        self.lambda_es = float(lambda_es)
        self.lambda_order = float(lambda_order)
        self.lambda_sign = float(lambda_sign)
        self.q_loss = QuantileLoss(alpha)
    def forward(self, y, q, e):
        loss_var = self.q_loss(y, q)
        hit = (y <= q).float()
        es_sq = hit * (e - y) ** 2
        loss_es = es_sq.sum() / hit.sum().clamp_min(1.0)
        loss_order = F.relu(e - q).mean()
        loss_sign = (F.relu(q) + F.relu(e)).mean()
        return self.lambda_var * loss_var + self.lambda_es * loss_es + self.lambda_order * loss_order + self.lambda_sign * loss_sign

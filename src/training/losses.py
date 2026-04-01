from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    """
    Pinball loss for VaR forecast.
    """
    def __init__(self, alpha: float) -> None:
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)

    def forward(self, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        u = y - q
        return torch.maximum(self.alpha * u, (self.alpha - 1.0) * u).mean()


class StableVaRESLoss(nn.Module):
    """
    Stable surrogate loss for joint VaR / ES training.

    Components:
      1. Quantile (pinball) loss for VaR
      2. ES regression toward tail outcomes on exceedance points
      3. Soft constraint penalty enforcing ES <= VaR
      4. Soft penalty encouraging negative left-tail forecasts

    This is intentionally more stable than the previous FZ-like loss.
    """
    def __init__(
        self,
        alpha: float,
        lambda_var: float = 1.0,
        lambda_es: float = 1.0,
        lambda_order: float = 10.0,
        lambda_sign: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.lambda_var = float(lambda_var)
        self.lambda_es = float(lambda_es)
        self.lambda_order = float(lambda_order)
        self.lambda_sign = float(lambda_sign)
        self.eps = float(eps)
        self.q_loss = QuantileLoss(alpha)

    def forward(self, y: torch.Tensor, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # 1) VaR calibration
        loss_var = self.q_loss(y, q)

        # 2) ES target on exceedances only
        hit = (y <= q).float()

        # When hit=1, target ES should track actual tail return y.
        # When no exceedances occur in a batch, denominator is protected.
        es_sq = hit * (e - y) ** 2
        loss_es = es_sq.sum() / hit.sum().clamp_min(1.0)

        # 3) Structural constraint: ES <= VaR
        loss_order = F.relu(e - q).mean()

        # 4) Left-tail forecasts should stay negative
        loss_sign = (F.relu(q) + F.relu(e)).mean()

        total = (
            self.lambda_var * loss_var
            + self.lambda_es * loss_es
            + self.lambda_order * loss_order
            + self.lambda_sign * loss_sign
        )
        return total
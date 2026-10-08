"""
Non-linear fusion operators for combining classical and quantum branch logits.

Each operator takes (c_logits, q_logits) → fused_logits, all shape (B, num_classes).

Operators
---------
concat_linear  Baseline: cat → Linear(2*nc, nc)
sum            Equal-weight average (no parameters)
weighted       Learnable scalar α: α·c + (1−α)·q
product        Kittler product rule in log-prob space (output is log-prob → use NLLLoss)
gated          Input-dependent gate: g=σ(MLP(cat(c,q))), fused=g·c + (1−g)·q
max            Element-wise max of logits

The `is_log_prob` flag signals that a fusion op outputs log-probabilities
(not raw logits); the training script uses NLLLoss instead of CrossEntropyLoss
when this flag is set.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatLinearFusion(nn.Module):
    """Baseline: concatenate both logit vectors, project back to num_classes."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(num_classes * 2, num_classes)

    def forward(self, c_logits, q_logits):
        return self.linear(torch.cat([c_logits, q_logits], dim=1))


class SumFusion(nn.Module):
    """Equal-weight average of logits (no learnable parameters)."""

    def forward(self, c_logits, q_logits):
        return (c_logits + q_logits) * 0.5


class WeightedFusion(nn.Module):
    """Learnable scalar α (shared across classes): α·c + (1−α)·q."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))  # logit of 0.5

    def forward(self, c_logits, q_logits):
        alpha = torch.sigmoid(self.alpha)
        return alpha * c_logits + (1.0 - alpha) * q_logits


class ProductFusion(nn.Module):
    """
    Kittler product rule:
      log P_fused(k|x) = log P_c(k|x) + log P_q(k|x)   (+ normalisation const)

    The constant cancels when using NLLLoss, so the raw sum of log-softmaxes is
    returned directly.  Set is_log_prob=True so the training loop uses NLLLoss.
    """
    is_log_prob: bool = True

    def forward(self, c_logits, q_logits):
        return F.log_softmax(c_logits, dim=1) + F.log_softmax(q_logits, dim=1)


class GatedFusion(nn.Module):
    """
    Input-dependent gate: g = σ(MLP(cat(c, q))) ∈ (0,1)^{num_classes}.
    fused = g · c + (1 − g) · q.
    The gate network is small (2 linear layers) to keep the quantum signal visible.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_classes * 2, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, c_logits, q_logits):
        gate = self.gate_net(torch.cat([c_logits, q_logits], dim=1))
        return gate * c_logits + (1.0 - gate) * q_logits


class MaxFusion(nn.Module):
    """Element-wise max of logits (no learnable parameters)."""

    def forward(self, c_logits, q_logits):
        return torch.maximum(c_logits, q_logits)


# ── Registry ──────────────────────────────────────────────────────────────────

_FUSION_REGISTRY = {
    "concat_linear": ConcatLinearFusion,
    "sum":           SumFusion,
    "weighted":      WeightedFusion,
    "product":       ProductFusion,
    "gated":         GatedFusion,
    "max":           MaxFusion,
}

FUSION_NAMES = list(_FUSION_REGISTRY.keys())


def get_fusion(name: str, num_classes: int = 2) -> nn.Module:
    if name not in _FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion '{name}'. Choose from: {FUSION_NAMES}")
    cls = _FUSION_REGISTRY[name]
    try:
        return cls(num_classes=num_classes)
    except TypeError:
        return cls()

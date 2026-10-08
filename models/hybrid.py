"""
Hybrid quantum-classical model and classical-only ablation baseline.

HybridModel
-----------
Classical branch : backbone → 512/1024-d features → fc → c_logits
Quantum branch   : features → VQCBlock (proj + VQC) → q_feats → fc → q_logits
Fusion           : (c_logits, q_logits) → fused_logits   [any FusionOp]

forward() returns a 5-tuple so callers can access intermediate representations
for loss computation, Mahalanobis scoring, and explainability:
    (fused_logits, c_feats, q_feats, c_logits, q_logits)

ClassicalOnlyModel
------------------
Ablation baseline that wraps a backbone and exposes the same 5-tuple interface
(q_feats and q_logits are None). Used to isolate the quantum contribution.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from models.backbones import get_backbone
from models.vqc import VQCBlock
from models.fusion import get_fusion


class HybridModel(nn.Module):
    def __init__(self, backbone: nn.Module, vqc_block: nn.Module,
                 fusion: nn.Module, num_classes: int = 2):
        super().__init__()
        self.backbone  = backbone
        self.vqc_block = vqc_block          # VQCBlock: features → q_feats
        self.q_head    = nn.Linear(vqc_block.n_qubits, num_classes)
        self.fusion    = fusion
        self.num_classes = num_classes

    def forward(self, x):
        c_feats, c_logits = self.backbone(x)
        q_feats           = self.vqc_block(c_feats)     # (B, n_qubits)
        q_logits          = self.q_head(q_feats)
        fused_logits      = self.fusion(c_logits, q_logits)
        return fused_logits, c_feats, q_feats, c_logits, q_logits

    @property
    def feature_dim(self) -> int:
        """Joint embedding dim for Mahalanobis scoring (classical + quantum)."""
        return self.backbone.feature_dim + self.vqc_block.n_qubits

    def get_joint_embedding(self, x) -> torch.Tensor:
        """
        Concatenated [classical_features, quantum_features] for anomaly scoring.
        Gradients flow through by design — eval-only callers must wrap their own
        call site in torch.no_grad(); the adaptive PGD attack needs this
        differentiable to backprop the Mahalanobis loss into the input image.
        """
        _, c_feats, q_feats, _, _ = self.forward(x)
        return torch.cat([c_feats, q_feats], dim=1)

    def get_last_conv(self):
        return self.backbone.get_last_conv()


class ClassicalOnlyModel(nn.Module):
    """Ablation: same backbone, no VQC. 5-tuple interface: q_feats=q_logits=None."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        c_feats, c_logits = self.backbone(x)
        return c_logits, c_feats, None, c_logits, None

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    def get_joint_embedding(self, x) -> torch.Tensor:
        c_feats, _ = self.backbone(x)
        return c_feats

    def get_last_conv(self):
        return self.backbone.get_last_conv()


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(backbone_name: str,
                fusion_name: str = None,
                num_classes: int = 2,
                n_qubits: int = cfg.N_QUBITS,
                n_layers: int = cfg.VQC_LAYERS) -> nn.Module:
    """
    Build a model from its string identifiers.

    fusion_name=None  →  ClassicalOnlyModel (no VQC, no fusion layer)
    fusion_name=str   →  HybridModel with the specified fusion operator
    """
    backbone = get_backbone(backbone_name, num_classes=num_classes)
    if fusion_name is None:
        return ClassicalOnlyModel(backbone)
    vqc_block = VQCBlock(in_dim=backbone.feature_dim, n_qubits=n_qubits, n_layers=n_layers)
    fusion    = get_fusion(fusion_name, num_classes=num_classes)
    return HybridModel(backbone, vqc_block, fusion, num_classes=num_classes)


def model_key(backbone_name: str, fusion_name: str = None) -> str:
    """Stable string identifier for checkpoint paths and result tables."""
    return f"{backbone_name}__{fusion_name or 'classical'}"


def load_model_from_ckpt(ckpt: dict, device, num_classes: int = 2) -> nn.Module:
    """
    Rebuild + load a model from a checkpoint dict, using the VQC width/depth it
    was actually trained with (falls back to config defaults for checkpoints
    saved before n_qubits/n_layers were recorded). Every eval/explain script
    should use this instead of calling build_model() directly with a checkpoint,
    since a mismatched n_qubits/n_layers silently breaks state_dict loading.
    """
    model = build_model(
        ckpt["backbone"], ckpt.get("fusion"), num_classes=num_classes,
        n_qubits=ckpt.get("n_qubits", cfg.N_QUBITS),
        n_layers=ckpt.get("n_layers", cfg.VQC_LAYERS),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model

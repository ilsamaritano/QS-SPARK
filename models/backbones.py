"""
Backbone models for PLASyR-Q.

Both backbones:
- Accept 1-channel (grayscale) 224×224 spectrogram input
- Return (features, logits): penultimate embedding + classification head output
- Are ImageNet-pretrained with grayscale adaptation (average RGB first-layer weights)

SE-ResNet-34 is ported directly from PLASyR's CodeHPC/SE.py, keeping all
conv/bn weights from the pretrained ResNet-34 and randomly initialising the
new SE blocks (they have no pretrained equivalent).

DenseNet-121 uses timm's pretrained model with in_chans=1.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm
from torchvision import models
from torchvision.models import ResNet34_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── SE blocks (ported verbatim from PLASyR CodeHPC/SE.py) ────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample=None, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.se    = SEBlock(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ── SE-ResNet-34 backbone ─────────────────────────────────────────────────────

class SEResNet34Backbone(nn.Module):
    """
    SE-ResNet-34 adapted for grayscale SAR spectrograms.
    Returns (features_512d, logits_num_classes).
    Penultimate features (512d) are used downstream for:
      - VQC angle projection in HybridModel
      - Mahalanobis anomaly scoring in replay_eval
    """

    def __init__(self, num_classes: int = 2, reduction: int = 16):
        super().__init__()
        base = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(base, layer_name)
            setattr(base, layer_name, nn.Sequential(*[
                self._upgrade_block(b, reduction) for b in layer
            ]))

        # Average RGB conv1 weights → 1-channel grayscale
        old = base.conv1
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        base.conv1 = new

        self.stem    = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        self.fc      = nn.Linear(512, num_classes)

    @staticmethod
    def _upgrade_block(b, reduction: int) -> SEBasicBlock:
        se = SEBasicBlock(
            in_planes=b.conv1.in_channels,
            planes=b.conv1.out_channels,
            stride=b.conv1.stride[0],
            downsample=b.downsample,
            reduction=reduction,
        )
        se.conv1.load_state_dict(b.conv1.state_dict())
        se.bn1.load_state_dict(b.bn1.state_dict())
        se.conv2.load_state_dict(b.conv2.state_dict())
        se.bn2.load_state_dict(b.bn2.state_dict())
        return se

    @property
    def feature_dim(self) -> int:
        return 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)   # (B, 512)
        logits   = self.fc(features)     # (B, num_classes)
        return features, logits

    def get_last_conv(self) -> nn.Module:
        """Return the last conv layer — used as GradCAM target."""
        return self.layer4[-1].conv2


# ── DenseNet-121 backbone ─────────────────────────────────────────────────────

class DenseNet121Backbone(nn.Module):
    """
    DenseNet-121 with ImageNet weights, adapted to 1-channel input via timm.
    Returns (features_1024d, logits_num_classes).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # timm creates the model with in_chans=1 by averaging the RGB first-layer
        # weights, matching our grayscale adaptation strategy for SE-ResNet-34.
        self.features = timm.create_model(
            "densenet121", pretrained=True, in_chans=1, num_classes=0
        )
        self.fc = nn.Linear(1024, num_classes)

    @property
    def feature_dim(self) -> int:
        return 1024

    def forward(self, x):
        features = self.features(x)   # (B, 1024)
        logits   = self.fc(features)  # (B, num_classes)
        return features, logits

    def get_last_conv(self) -> nn.Module:
        """Return the last conv layer — used as GradCAM target."""
        return self.features.features.denseblock4.denselayer16.conv2


# ── Generic timm backbone (backbone benchmark) ───────────────────────────────

class TimmBackbone(nn.Module):
    """
    Any timm model, ImageNet-pretrained, adapted to 1-channel input via
    in_chans=1 (timm averages the RGB stem weights — same grayscale strategy
    as the two hand-built backbones). Returns (features, logits).
    """

    def __init__(self, timm_name: str, num_classes: int = 2):
        super().__init__()
        self.features = timm.create_model(
            timm_name, pretrained=True, in_chans=1, num_classes=0
        )
        # probe the actual pooled-embedding size: for some families (e.g.
        # mobilenetv3) the head outputs more channels than model.num_features
        with torch.no_grad():
            self._feature_dim = self.features(torch.zeros(1, 1, 224, 224)).shape[1]
        self.fc = nn.Linear(self._feature_dim, num_classes)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x):
        features = self.features(x)
        logits   = self.fc(features)
        return features, logits

    def get_last_conv(self) -> nn.Module:
        raise NotImplementedError(
            "Grad-CAM target not defined for generic timm backbones; "
            "explainability figures use se_resnet34/densenet121 only."
        )


# Benchmark backbones: canonical name -> timm model id
TIMM_BACKBONES = {
    "resnet18":        "resnet18",
    "resnet34":        "resnet34",            # plain ResNet-34: isolates the SE contribution
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3":     "mobilenetv3_large_100",
    "vit_small":       "vit_small_patch16_224",
}


# ── Factory ───────────────────────────────────────────────────────────────────

def get_backbone(name: str, num_classes: int = 2, reduction: int = 16) -> nn.Module:
    if name == "se_resnet34":
        return SEResNet34Backbone(num_classes=num_classes, reduction=reduction)
    if name == "densenet121":
        return DenseNet121Backbone(num_classes=num_classes)
    if name in TIMM_BACKBONES:
        return TimmBackbone(TIMM_BACKBONES[name], num_classes=num_classes)
    raise ValueError(f"Unknown backbone '{name}'. Choose from: se_resnet34, "
                     f"densenet121, {', '.join(TIMM_BACKBONES)}")

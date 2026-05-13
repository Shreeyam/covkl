"""Encoder architectures for CIFAR-scale self-supervised pretraining."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


class SmallConvNet(nn.Module):
    """Light ConvNet for 32x32 images, with a 3-layer MLP projection head."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def forward_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for 32x32 inputs (3x3 stride-1 stem, no maxpool)."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone  # outputs 512-d

        self.head = nn.Sequential(
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, embed_dim),
        )

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_encoder(arch: str, embed_dim: int = 256) -> nn.Module:
    if arch == "resnet18":
        return ResNet18CIFAR(embed_dim)
    if arch == "smallconv":
        return SmallConvNet(embed_dim)
    raise ValueError(f"unknown arch: {arch}")

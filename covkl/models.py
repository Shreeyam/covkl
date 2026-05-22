"""Encoder architectures for self-supervised pretraining."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torchvision.models as tvm


def _make_projector(
    in_dim: int,
    embed_dim: int,
    *,
    depth: int,
    hidden_dim: int,
    use_bn: bool = True,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError(f"projector depth must be >= 1, got {depth}")

    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, embed_dim))
    return nn.Sequential(*layers)


class SmallConvNet(nn.Module):
    """Light ConvNet for 32x32 images, with a 3-layer MLP projection head."""

    def __init__(
        self,
        embed_dim: int = 256,
        *,
        projector_depth: int = 3,
        projector_hidden_dim: int = 512,
        projector_bn: bool = True,
    ):
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
        self.head = _make_projector(
            256,
            embed_dim,
            depth=projector_depth,
            hidden_dim=projector_hidden_dim,
            use_bn=projector_bn,
        )

    def forward_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for 32x32 inputs (3x3 stride-1 stem, no maxpool)."""

    def __init__(
        self,
        embed_dim: int = 256,
        *,
        projector_depth: int = 3,
        projector_hidden_dim: int = 1024,
        projector_bn: bool = True,
    ):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone  # outputs 512-d

        self.head = _make_projector(
            512,
            embed_dim,
            depth=projector_depth,
            hidden_dim=projector_hidden_dim,
            use_bn=projector_bn,
        )

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


class ResNet18ImageNet(nn.Module):
    """Standard ResNet-18 stem for larger ImageNet-style inputs."""

    def __init__(
        self,
        embed_dim: int = 256,
        *,
        projector_depth: int = 3,
        projector_hidden_dim: int = 1024,
        projector_bn: bool = True,
    ):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = _make_projector(
            512,
            embed_dim,
            depth=projector_depth,
            hidden_dim=projector_hidden_dim,
            use_bn=projector_bn,
        )

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_encoder(
    arch: str,
    embed_dim: int = 256,
    *,
    projector_depth: int = 3,
    projector_hidden_dim: Optional[int] = None,
    projector_bn: bool = True,
) -> nn.Module:
    if arch == "resnet18":
        return ResNet18CIFAR(
            embed_dim,
            projector_depth=projector_depth,
            projector_hidden_dim=projector_hidden_dim or 1024,
            projector_bn=projector_bn,
        )
    if arch in {"resnet18_imagenet", "resnet18-in"}:
        return ResNet18ImageNet(
            embed_dim,
            projector_depth=projector_depth,
            projector_hidden_dim=projector_hidden_dim or 1024,
            projector_bn=projector_bn,
        )
    if arch == "smallconv":
        return SmallConvNet(
            embed_dim,
            projector_depth=projector_depth,
            projector_hidden_dim=projector_hidden_dim or 512,
            projector_bn=projector_bn,
        )
    raise ValueError(f"unknown arch: {arch}")

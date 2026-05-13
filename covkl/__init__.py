"""Gaussian-KL covariance matching for self-supervised learning on CIFAR-10."""

from covkl.models import ResNet18CIFAR, SmallConvNet, build_encoder
from covkl.train import train_and_eval

__all__ = ["ResNet18CIFAR", "SmallConvNet", "build_encoder", "train_and_eval"]

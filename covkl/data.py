"""CIFAR-10 two-view augmentations and dataloaders."""

from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class MultiCropCIFAR:
    """Two random crops of each CIFAR image (SimCLR-style augmentations)."""

    def __init__(self):
        self.transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


def collate_views(batch):
    v1 = torch.stack([b[0][0] for b in batch])
    v2 = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return (v1, v2), labels


def eval_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def build_loaders(data_root: str, batch_size: int = 256, num_workers: int = 4):
    """Build (train, eval_train, eval_test) CIFAR-10 dataloaders.

    Train uses two-view augmentation; eval loaders use deterministic transforms.
    """
    torchvision.datasets.CIFAR10(data_root, train=True, download=True)
    torchvision.datasets.CIFAR10(data_root, train=False, download=True)

    train_ds = torchvision.datasets.CIFAR10(data_root, train=True, transform=MultiCropCIFAR())
    eval_train = torchvision.datasets.CIFAR10(data_root, train=True, transform=eval_transform())
    eval_test = torchvision.datasets.CIFAR10(data_root, train=False, transform=eval_transform())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_views, drop_last=True, persistent_workers=num_workers > 0,
    )
    eval_train_loader = DataLoader(
        eval_train, batch_size=1024, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    eval_test_loader = DataLoader(
        eval_test, batch_size=1024, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, eval_train_loader, eval_test_loader

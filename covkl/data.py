"""Dataset transforms and dataloaders for two-view SSL training."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenet100": 100}


class TwoViewTransform:
    """Apply the same stochastic transform twice to produce SSL views."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class HFDataset(Dataset):
    """Thin torch Dataset wrapper around a HuggingFace image dataset split."""

    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        image = item["image"].convert("RGB")
        return self.transform(image), int(item["label"])


def collate_views(batch):
    v1 = torch.stack([b[0][0] for b in batch])
    v2 = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return (v1, v2), labels


def num_classes_for_dataset(dataset: str) -> int:
    dataset = _normalize_dataset_name(dataset)
    return NUM_CLASSES[dataset]


def train_transform(dataset: str, image_size: Optional[int] = None) -> T.Compose:
    dataset = _normalize_dataset_name(dataset)
    if dataset in {"cifar10", "cifar100"}:
        size = image_size or 32
        mean, std = CIFAR_MEAN, CIFAR_STD
        scale = (0.5, 1.0)
    else:
        size = image_size or 160
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        scale = (0.2, 1.0)
    return T.Compose([
        T.RandomResizedCrop(size, scale=scale),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transform(dataset: str, image_size: Optional[int] = None) -> T.Compose:
    dataset = _normalize_dataset_name(dataset)
    if dataset in {"cifar10", "cifar100"}:
        transforms = []
        if image_size and image_size != 32:
            transforms.append(T.Resize(image_size))
        transforms.extend([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
        return T.Compose(transforms)

    size = image_size or 160
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 4,
    *,
    dataset: str = "cifar10",
    image_size: Optional[int] = None,
    eval_batch_size: int = 1024,
    seed: Optional[int] = None,
):
    """Build (train, eval_train, eval_test) dataloaders.

    Train uses two-view augmentation; eval loaders use deterministic transforms.
    """
    dataset = _normalize_dataset_name(dataset)
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker

    if dataset == "cifar10":
        train_ds, eval_train, eval_test = _build_cifar(
            torchvision.datasets.CIFAR10, data_root, dataset, image_size,
        )
    elif dataset == "cifar100":
        train_ds, eval_train, eval_test = _build_cifar(
            torchvision.datasets.CIFAR100, data_root, dataset, image_size,
        )
    elif dataset == "imagenet100":
        train_ds, eval_train, eval_test = _build_imagenet100(data_root, image_size)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_views, drop_last=True, persistent_workers=num_workers > 0,
        generator=generator, worker_init_fn=worker_init_fn,
    )
    eval_train_loader = DataLoader(
        eval_train, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0, worker_init_fn=worker_init_fn,
    )
    eval_test_loader = DataLoader(
        eval_test, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0, worker_init_fn=worker_init_fn,
    )
    return train_loader, eval_train_loader, eval_test_loader


def _build_cifar(dataset_cls, data_root: str, dataset: str, image_size: Optional[int]):
    dataset_cls(data_root, train=True, download=True)
    dataset_cls(data_root, train=False, download=True)
    train_ds = dataset_cls(
        data_root, train=True, transform=TwoViewTransform(train_transform(dataset, image_size)),
    )
    eval_train = dataset_cls(data_root, train=True, transform=eval_transform(dataset, image_size))
    eval_test = dataset_cls(data_root, train=False, transform=eval_transform(dataset, image_size))
    return train_ds, eval_train, eval_test


def _build_imagenet100(data_root: str, image_size: Optional[int]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "ImageNet-100 loading requires the `datasets` package. "
            "Install project dependencies from pyproject.toml."
        ) from exc

    root = _imagenet100_root(data_root)
    cache_dir = root / ".cache" / "hf_datasets"
    os.makedirs(cache_dir, exist_ok=True)
    train_split = load_dataset(str(root), split="train", cache_dir=str(cache_dir))
    val_split = load_dataset(str(root), split="validation", cache_dir=str(cache_dir))
    train_ds = HFDataset(train_split, TwoViewTransform(train_transform("imagenet100", image_size)))
    eval_train = HFDataset(train_split, eval_transform("imagenet100", image_size))
    eval_test = HFDataset(val_split, eval_transform("imagenet100", image_size))
    return train_ds, eval_train, eval_test


def _imagenet100_root(data_root: str) -> Path:
    root = Path(data_root)
    if (root / "data").exists() and (root / "README.md").exists():
        return root
    candidate = root / "imagenet-100"
    if (candidate / "data").exists():
        return candidate
    raise FileNotFoundError(
        "Could not find ImageNet-100. Expected either DATA_ROOT to point to "
        "the dataset directory or DATA_ROOT/imagenet-100 to exist."
    )


def _normalize_dataset_name(dataset: str) -> str:
    normalized = dataset.lower().replace("_", "-")
    aliases = {
        "cifar-10": "cifar10",
        "cifar10": "cifar10",
        "cifar-100": "cifar100",
        "cifar100": "cifar100",
        "imagenet-100": "imagenet100",
        "imagenet100": "imagenet100",
    }
    if normalized not in aliases:
        raise ValueError(f"unknown dataset: {dataset}. Available: {sorted(NUM_CLASSES)}")
    return aliases[normalized]


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

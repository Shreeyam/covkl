"""UMAP of a saved mini_experiment checkpoint."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import umap
from torch.utils.data import DataLoader

from flat_dino.mini_experiment import ResNet18CIFAR, SmallConvNet


@torch.no_grad()
def extract_head_features(model, loader, device):
    model.eval()
    feats, labs = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).cpu())
        labs.append(y)
    return torch.cat(feats).numpy(), torch.cat(labs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default="./results/mini")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "mps" if (torch.backends.mps.is_available() and not args.cpu) else "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"Loaded checkpoint at epoch {ckpt['epoch']} with config: {cfg}")

    arch = cfg.get("arch", "smallconv")
    model_cls = ResNet18CIFAR if arch == "resnet18" else SmallConvNet
    model = model_cls(embed_dim=256).to(device)
    model.load_state_dict(ckpt["student"])
    model.eval()

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    val = torchvision.datasets.CIFAR10("./data", train=False, download=False,
                                        transform=eval_transform)
    loader = DataLoader(val, batch_size=1024, shuffle=False, num_workers=4)

    print("Extracting head features...")
    feats, labels = extract_head_features(model, loader, device)

    print(f"Fitting UMAP on {feats.shape[0]} points in {feats.shape[1]}-d...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedded = reducer.fit_transform(feats)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.tab10
    cifar_classes = ["airplane", "auto", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
    for c in range(10):
        mask = labels == c
        axes[0].scatter(embedded[mask, 0], embedded[mask, 1], s=2, alpha=0.5,
                        c=[cmap(c)], label=cifar_classes[c])
    axes[0].set_title(f"{tag} @ epoch {ckpt['epoch']} — UMAP")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].legend(fontsize=7, markerscale=3, loc="best")

    norms = np.linalg.norm(feats, axis=1)
    axes[1].hist(norms, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    axes[1].set_title(f"Feature norms: μ={norms.mean():.2f}  σ={norms.std():.2f}")
    axes[1].set_xlabel("L2 norm")
    axes[1].axvline(norms.mean(), color="red", linestyle="--", linewidth=1)

    fig.tight_layout()
    out = os.path.join(args.out_dir, f"umap_{tag}_ep{ckpt['epoch']}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

"""PC1/PC2 scatter of CIFAR-10 validation features, coloured by class."""

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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from flat_dino.mini_experiment import ResNet18CIFAR


CIFAR_CLASSES = ["airplane", "auto", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def get_head_features(model, loader, device):
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

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ResNet18CIFAR(embed_dim=256).to(device)
    model.load_state_dict(ckpt["student"])
    model.eval()

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    val = torchvision.datasets.CIFAR10("./data", train=False, download=False,
                                        transform=eval_transform)
    loader = DataLoader(val, batch_size=1024, shuffle=False, num_workers=4)
    feats, labels = get_head_features(model, loader, device)
    print(f"Features: {feats.shape}")

    # Center (should already be ~ zero mean thanks to training)
    mean = feats.mean(axis=0)
    print(f"Empirical mean norm: {np.linalg.norm(mean):.3f}")
    centered = feats - mean
    pca = PCA(n_components=2)
    proj = pca.fit_transform(centered)
    evr = pca.explained_variance_ratio_
    print(f"PCA EVR (top 2): {evr[0]:.4f}, {evr[1]:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Clip to 99th-percentile extent so the bulk of data is visible
    clip = np.quantile(np.abs(proj), 0.99)

    # Also compute LDA (supervised) projection for comparison
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_proj = lda.fit_transform(centered, labels)
    print(f"LDA explained variance: {lda.explained_variance_ratio_}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    cmap = plt.cm.tab10

    for ax, data, name, clip_val in [
        (axes[0], proj, f"PCA: PC1 ({evr[0]:.3f})  vs  PC2 ({evr[1]:.3f})", clip),
        (axes[1], lda_proj, "LDA: top 2 discriminant directions", None),
    ]:
        for c in range(10):
            mask = labels == c
            ax.scatter(data[mask, 0], data[mask, 1], s=5, alpha=0.45,
                       c=[cmap(c)], label=CIFAR_CLASSES[c], edgecolors="none")
        for c in range(10):
            mask = labels == c
            cx, cy = data[mask].mean(axis=0)
            ax.scatter(cx, cy, s=180, c=[cmap(c)], edgecolors="black",
                       linewidths=1.5, zorder=6, marker="D")
        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
        ax.scatter(0, 0, s=120, marker="+", color="black", zorder=10)
        ax.set_aspect("equal")
        ax.set_title(name)

        if clip_val is not None:
            ax.set_xlim(-clip_val, clip_val)
            ax.set_ylim(-clip_val, clip_val)

    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("LD1"); axes[1].set_ylabel("LD2")
    axes[0].legend(fontsize=7, ncol=2, markerscale=2, loc="upper right",
                   framealpha=0.9)
    fig.suptitle(f"CIFAR-10 val features (ep {ckpt['epoch']}) — "
                 f"dots = samples, diamonds = class centroids", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"pca_scatter_ep{ckpt['epoch']}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

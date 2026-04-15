"""Project noise-corruption trajectories into a 2-D PCA plane and
display them as polar / 2-D scatter curves."""

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

from flat_dino.mini_experiment import ResNet18CIFAR


CIFAR_CLASSES = ["airplane", "auto", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def embed(model, imgs, device):
    return model(imgs.to(device)).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--n-alphas", type=int, default=21)
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

    mean_t = torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
    std_t = torch.tensor([0.2470, 0.2435, 0.2616])[:, None, None]

    # Sample images per class
    to_tensor = T.ToTensor()
    raw_ds = torchvision.datasets.CIFAR10("./data", train=False, download=False)
    class_imgs = {c: [] for c in range(10)}
    for i in range(len(raw_ds)):
        img, lbl = raw_ds[i]
        if len(class_imgs[lbl]) < args.n_per_class:
            class_imgs[lbl].append(to_tensor(img))
        if all(len(v) >= args.n_per_class for v in class_imgs.values()):
            break

    all_imgs = torch.stack([img for c in range(10) for img in class_imgs[c]])
    all_labels = np.array([c for c in range(10) for _ in range(args.n_per_class)])
    N = len(all_imgs)

    torch.manual_seed(0)
    noise = torch.rand_like(all_imgs)
    alphas = np.linspace(0.0, 1.0, args.n_alphas)

    # Embed all (sample × alpha) combos
    print(f"Embedding {N} samples × {args.n_alphas} alphas = {N * args.n_alphas} total...")
    feats = np.zeros((args.n_alphas, N, 256))
    for ai, a in enumerate(alphas):
        mixed = (1 - a) * all_imgs + a * noise
        normalized = (mixed - mean_t) / std_t
        feats[ai] = embed(model, normalized, device)

    # PCA on clean features (α=0)
    clean = feats[0]
    pca = PCA(n_components=3)
    pca.fit(clean)
    print(f"PCA explained variance (clean): {pca.explained_variance_ratio_}")

    # Project every (α, sample) embedding into PCA plane
    flat = feats.reshape(-1, 256)
    proj_all = pca.transform(flat).reshape(args.n_alphas, N, 3)

    norms = np.linalg.norm(feats, axis=-1)  # (n_alphas, N)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"ep{ckpt['epoch']}"

    # === Plot 1: 2-D PC scatter with trajectories ===
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.cm.tab10
    for i in range(N):
        c = all_labels[i]
        # Trajectory
        ax.plot(proj_all[:, i, 0], proj_all[:, i, 1],
                "-", color=cmap(c), alpha=0.25, linewidth=0.8)
        # Clean endpoint
        ax.scatter(proj_all[0, i, 0], proj_all[0, i, 1],
                   s=30, color=cmap(c), edgecolors="black", linewidth=0.5, zorder=5)
        # Noise endpoint
        ax.scatter(proj_all[-1, i, 0], proj_all[-1, i, 1],
                   s=10, color=cmap(c), alpha=0.6, marker="x", zorder=4)

    # Origin marker
    ax.scatter(0, 0, s=120, marker="+", color="black", zorder=10, label="origin")
    for c in range(10):
        ax.scatter([], [], color=cmap(c), label=CIFAR_CLASSES[c], s=30)

    ax.axhline(0, color="gray", linewidth=0.3, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
    ax.set_title(f"Noise trajectories in PC1-PC2 plane (ep {ckpt['epoch']})\n"
                 "● clean  ✕ pure noise")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, f"noise_pca_scatter_{tag}.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # === Plot 2: Polar plot — angle from PC1-PC2 plane, radius = true norm ===
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    for i in range(N):
        c = all_labels[i]
        theta = np.arctan2(proj_all[:, i, 1], proj_all[:, i, 0])
        r = norms[:, i]
        ax.plot(theta, r, "-", color=cmap(c), alpha=0.3, linewidth=0.8)
        ax.scatter(theta[0], r[0], s=30, color=cmap(c), edgecolors="black",
                   linewidth=0.5, zorder=5)
        ax.scatter(theta[-1], r[-1], s=10, color=cmap(c), alpha=0.6,
                   marker="x", zorder=4)

    for c in range(10):
        ax.scatter([], [], color=cmap(c), label=CIFAR_CLASSES[c], s=30)

    ax.set_title(f"Polar: angle in PC1-PC2 plane, radius = true norm (ep {ckpt['epoch']})",
                 pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, f"noise_polar_{tag}.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # === Plot 3: Radius vs α, coloured by class ===
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for i in range(N):
        c = all_labels[i]
        ax.plot(alphas, norms[:, i], "-", color=cmap(c), alpha=0.25, linewidth=0.7)
    # Class means
    for c in range(10):
        mask = all_labels == c
        ax.plot(alphas, norms[:, mask].mean(axis=1), "-", color=cmap(c),
                linewidth=2.5, label=CIFAR_CLASSES[c])
    ax.set_xlabel("Noise fraction α")
    ax.set_ylabel("Feature norm")
    ax.set_title(f"Per-sample (thin) and per-class mean (thick) norm vs α (ep {ckpt['epoch']})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out3 = os.path.join(args.out_dir, f"noise_radius_{tag}.png")
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()

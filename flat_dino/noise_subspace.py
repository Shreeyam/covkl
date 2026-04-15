"""Analyze the 'noise subspace' — where in embedding space do noise inputs live?

Generates:
  - UMAP of real CIFAR samples vs pure-noise embeddings
  - Effective dimensionality of noise embeddings via PCA spectrum
  - Angular relationship between noise cluster and class centers
"""

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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from flat_dino.mini_experiment import ResNet18CIFAR


CIFAR_CLASSES = ["airplane", "auto", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def embed(model, imgs, device):
    return model(imgs.to(device)).cpu().numpy()


@torch.no_grad()
def embed_loader(model, loader, device):
    feats, labs = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).cpu())
        labs.append(y)
    return torch.cat(feats).numpy(), torch.cat(labs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-real", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=500)
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

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    val_ds = torchvision.datasets.CIFAR10("./data", train=False, download=False,
                                           transform=eval_transform)
    # Subsample n_real real images
    rng = np.random.RandomState(42)
    idx = rng.choice(len(val_ds), args.n_real, replace=False)
    subset = torch.utils.data.Subset(val_ds, idx.tolist())
    loader = DataLoader(subset, batch_size=1024, shuffle=False, num_workers=4)

    print(f"Extracting {args.n_real} real image features...")
    real_feats, real_labels = embed_loader(model, loader, device)

    print(f"Generating {args.n_noise} pure-noise embeddings...")
    torch.manual_seed(0)
    noise_batches = []
    chunk = 200
    for i in range(0, args.n_noise, chunk):
        n = min(chunk, args.n_noise - i)
        raw_noise = torch.rand(n, 3, 32, 32)  # uniform [0, 1]
        normalized = (raw_noise - mean_t) / std_t
        noise_batches.append(embed(model, normalized, device))
    noise_feats = np.concatenate(noise_batches, axis=0)

    # === 1. PCA spectrum of noise vs real ===
    real_pca = PCA()
    real_pca.fit(real_feats - real_feats.mean(axis=0))
    noise_pca = PCA()
    noise_pca.fit(noise_feats - noise_feats.mean(axis=0))

    # Effective dim = exp(entropy(spectrum))
    def eff_dim(var_ratio):
        p = var_ratio / var_ratio.sum()
        p = p[p > 0]
        return np.exp(-(p * np.log(p)).sum())

    print(f"\nEffective dimensionality (entropy of PCA spectrum):")
    print(f"  Real:  {eff_dim(real_pca.explained_variance_ratio_):.1f} / 256")
    print(f"  Noise: {eff_dim(noise_pca.explained_variance_ratio_):.1f} / 256")

    # === 2. Centers and distances ===
    class_centers = np.stack([real_feats[real_labels == c].mean(axis=0) for c in range(10)])
    noise_center = noise_feats.mean(axis=0)

    # Angles between noise center and each class center
    def unit(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    u_noise = unit(noise_center)
    u_classes = unit(class_centers)
    cos_sims = u_classes @ u_noise  # (10,)
    angles_deg = np.degrees(np.arccos(np.clip(cos_sims, -1, 1)))

    print(f"\nNoise-center ‖‖ = {np.linalg.norm(noise_center):.3f}")
    print(f"Noise cluster spread: mean dist to noise center = "
          f"{np.linalg.norm(noise_feats - noise_center, axis=1).mean():.3f}")
    print(f"\nAngle (deg) from noise center to each class center:")
    for c in range(10):
        print(f"  {CIFAR_CLASSES[c]:<10} {angles_deg[c]:>6.1f}°  "
              f"(‖class_center‖={np.linalg.norm(class_centers[c]):.2f})")
    mean_angle_classes = np.mean([np.degrees(np.arccos(np.clip(
        u_classes[i] @ u_classes[j], -1, 1)))
        for i in range(10) for j in range(i + 1, 10)])
    print(f"Mean pairwise angle between class centers: {mean_angle_classes:.1f}°")

    # === 3. UMAP of real + noise ===
    print(f"\nFitting UMAP on {args.n_real} real + {args.n_noise} noise = "
          f"{args.n_real + args.n_noise} points...")
    combined = np.concatenate([real_feats, noise_feats], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30,
                        min_dist=0.05)
    embedded = reducer.fit_transform(combined)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"ep{ckpt['epoch']}"

    # Plot 1: UMAP
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    cmap = plt.cm.tab10
    real_emb = embedded[:args.n_real]
    noise_emb = embedded[args.n_real:]
    for c in range(10):
        mask = real_labels == c
        ax.scatter(real_emb[mask, 0], real_emb[mask, 1],
                   s=4, alpha=0.5, c=[cmap(c)], label=CIFAR_CLASSES[c])
    ax.scatter(noise_emb[:, 0], noise_emb[:, 1],
               s=20, alpha=0.8, c="black", marker="x", linewidths=1,
               label="pure noise", zorder=10)
    ax.set_title(f"UMAP: real CIFAR vs pure noise embeddings (ep {ckpt['epoch']})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, ncol=2, markerscale=2)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, f"noise_subspace_umap_{tag}.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # Plot 2: PCA spectrum comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    k = 40
    axes[0].plot(real_pca.explained_variance_ratio_[:k], "o-", label="real", markersize=3)
    axes[0].plot(noise_pca.explained_variance_ratio_[:k], "s-", label="noise", markersize=3)
    axes[0].set_xlabel("PC index")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("PCA spectrum (top 40 PCs)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    axes[1].plot(np.cumsum(real_pca.explained_variance_ratio_[:k]), "o-",
                 label="real", markersize=3)
    axes[1].plot(np.cumsum(noise_pca.explained_variance_ratio_[:k]), "s-",
                 label="noise", markersize=3)
    axes[1].set_xlabel("PC index")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("Cumulative variance (top 40 PCs)")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0.9, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle(f"Effective dim: real ≈ {eff_dim(real_pca.explained_variance_ratio_):.0f}, "
                 f"noise ≈ {eff_dim(noise_pca.explained_variance_ratio_):.0f} / 256")
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, f"noise_subspace_pca_{tag}.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # Plot 3: Angles as a bar chart
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    order = np.argsort(angles_deg)
    ax.bar([CIFAR_CLASSES[c] for c in order], angles_deg[order],
           color=[cmap(c) for c in order], edgecolor="black")
    ax.axhline(90, color="red", linestyle="--", alpha=0.5, label="90° (orthogonal)")
    ax.axhline(mean_angle_classes, color="gray", linestyle=":", alpha=0.7,
               label=f"mean class-class = {mean_angle_classes:.1f}°")
    ax.set_ylabel("Angle to noise center (deg)")
    ax.set_title(f"How far from noise direction is each class center? (ep {ckpt['epoch']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out3 = os.path.join(args.out_dir, f"noise_subspace_angles_{tag}.png")
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()

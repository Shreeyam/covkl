"""Investigate what the feature norm encodes beyond class identity."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from scipy import stats
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


def image_stats(raw_imgs):
    """Compute simple per-image statistics from raw uint8 images (N, H, W, 3)."""
    imgs = raw_imgs.astype(np.float32) / 255.0
    mean = imgs.mean(axis=(1, 2, 3))
    std = imgs.std(axis=(1, 2, 3))
    # Per-channel color variance (color richness)
    channel_means = imgs.mean(axis=(1, 2))  # (N, 3)
    saturation = channel_means.std(axis=1)
    # Entropy from grayscale histogram
    gray = imgs.mean(axis=3)  # (N, H, W)
    entropy = np.zeros(len(imgs))
    for i in range(len(imgs)):
        hist, _ = np.histogram(gray[i], bins=32, range=(0, 1), density=False)
        p = hist / (hist.sum() + 1e-8)
        p = p[p > 0]
        entropy[i] = -(p * np.log(p)).sum()
    return {"mean": mean, "std": std, "saturation": saturation, "entropy": entropy}


def knn_per_sample(train_f, train_l, test_f, test_l, k=20):
    """Per-sample kNN correct/incorrect flag."""
    train_t = torch.from_numpy(train_f); test_t = torch.from_numpy(test_f)
    train_t = F.normalize(train_t, dim=-1); test_t = F.normalize(test_t, dim=-1)

    correct = np.zeros(len(test_l), dtype=bool)
    preds = np.zeros(len(test_l), dtype=int)
    for i in range(0, len(test_t), 1024):
        chunk = test_t[i:i + 1024]
        sim = chunk @ train_t.T
        _, topk_idx = sim.topk(k, dim=-1)
        topk_labels = torch.from_numpy(train_l)[topk_idx]
        votes = torch.zeros(chunk.shape[0], 10)
        for c in range(10):
            votes[:, c] = (topk_labels == c).float().sum(dim=-1)
        pred = votes.argmax(dim=-1).numpy()
        preds[i:i + len(chunk)] = pred
        correct[i:i + len(chunk)] = pred == test_l[i:i + len(chunk)]
    return correct, preds


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
    val_tensor = torchvision.datasets.CIFAR10("./data", train=False, download=False,
                                               transform=eval_transform)
    train_tensor = torchvision.datasets.CIFAR10("./data", train=True, download=False,
                                                 transform=eval_transform)
    val_raw = torchvision.datasets.CIFAR10("./data", train=False, download=False)
    val_imgs = np.stack([np.array(img) for img, _ in val_raw])

    val_loader = DataLoader(val_tensor, batch_size=1024, shuffle=False, num_workers=4)
    train_loader = DataLoader(train_tensor, batch_size=1024, shuffle=False, num_workers=4)

    print("Extracting features...")
    val_f, val_l = get_head_features(model, val_loader, device)
    train_f, train_l = get_head_features(model, train_loader, device)

    norms = np.linalg.norm(val_f, axis=1)

    # 1. Distance from (empirical) class center
    centers = np.stack([val_f[val_l == c].mean(axis=0) for c in range(10)])
    dist_from_center = np.array([np.linalg.norm(val_f[i] - centers[val_l[i]])
                                  for i in range(len(val_f))])

    # 2. kNN correctness per sample
    print("Running per-sample kNN...")
    knn_correct, knn_preds = knn_per_sample(train_f, train_l, val_f, val_l, k=20)
    print(f"Overall kNN acc: {knn_correct.mean():.4f}")

    # 3. Raw image stats
    print("Computing image statistics...")
    stats_img = image_stats(val_imgs)

    # === Correlations ===
    print("\n=== Correlations of norm with various quantities (Pearson r) ===")
    rows = [
        ("Intra-class distance from center", dist_from_center),
        ("kNN correct (1/0)",                 knn_correct.astype(float)),
        ("Raw image mean intensity",          stats_img["mean"]),
        ("Raw image std (contrast)",          stats_img["std"]),
        ("Color saturation",                  stats_img["saturation"]),
        ("Grayscale entropy",                 stats_img["entropy"]),
    ]
    print(f"{'Quantity':<38} {'r':>7} {'p':>10}")
    for name, vec in rows:
        r, p = stats.pearsonr(norms, vec)
        print(f"{name:<38} {r:>+7.3f} {p:>10.2e}")

    # === Plot 1: scatter of norm vs each quantity ===
    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (name, vec) in zip(axes.flat, rows):
        ax.scatter(vec, norms, s=2, alpha=0.3, c=val_l, cmap="tab10")
        r, _ = stats.pearsonr(norms, vec)
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel("L2 norm", fontsize=9)
        ax.set_title(f"r = {r:+.3f}", fontsize=10)
    fig.suptitle(f"What does feature norm encode? (ep {ckpt['epoch']}, 256-d head)",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"norm_investigation_ep{ckpt['epoch']}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # === Plot 2: highest/lowest norm images overall ===
    top = np.argsort(norms)[::-1][:16]
    bot = np.argsort(norms)[:16]
    fig, axes = plt.subplots(2, 16, figsize=(20, 3))
    for i, (idx_list, label) in enumerate([(top, "High norm"), (bot, "Low norm")]):
        for j, idx in enumerate(idx_list):
            axes[i, j].imshow(val_imgs[idx])
            title = f"{CIFAR_CLASSES[val_l[idx]]}\n‖z‖={norms[idx]:.1f}"
            if j == 0:
                title = label + ":\n" + title
            axes[i, j].set_title(title, fontsize=7)
            axes[i, j].axis("off")
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, f"norm_extremes_ep{ckpt['epoch']}.png")
    fig.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # === Plot 3: kNN accuracy vs norm quantile ===
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # Bin by norm quantile
    nbins = 10
    quantiles = np.quantile(norms, np.linspace(0, 1, nbins + 1))
    bin_centers, bin_acc = [], []
    for i in range(nbins):
        mask = (norms >= quantiles[i]) & (norms < quantiles[i + 1] + 1e-6)
        if mask.sum() > 0:
            bin_centers.append(norms[mask].mean())
            bin_acc.append(knn_correct[mask].mean())
    ax.plot(bin_centers, bin_acc, "o-", markersize=6)
    ax.set_xlabel("Feature norm (binned)")
    ax.set_ylabel("kNN accuracy")
    ax.set_title(f"kNN accuracy vs norm magnitude (ep {ckpt['epoch']})")
    ax.axhline(knn_correct.mean(), color="gray", linestyle="--",
               label=f"overall = {knn_correct.mean():.3f}")
    ax.legend()
    fig.tight_layout()
    out3 = os.path.join(args.out_dir, f"norm_vs_accuracy_ep{ckpt['epoch']}.png")
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()

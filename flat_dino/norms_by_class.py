"""Analyze whether the encoder uses different norm magnitudes per class."""

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

    norms = np.linalg.norm(feats, axis=1)
    print(f"\nOverall norm: μ={norms.mean():.3f}  σ={norms.std():.3f}\n")

    # Per-class statistics
    print(f"{'Class':<10} {'n':>4} {'μ_norm':>8} {'σ_norm':>8} {'min':>6} {'max':>6}")
    per_class = {}
    for c in range(10):
        n = norms[labels == c]
        per_class[c] = n
        print(f"{CIFAR_CLASSES[c]:<10} {len(n):>4} {n.mean():>8.3f} "
              f"{n.std():>8.3f} {n.min():>6.2f} {n.max():>6.2f}")

    # Rank classes by mean norm
    means = np.array([per_class[c].mean() for c in range(10)])
    rank = np.argsort(means)[::-1]
    print("\nRanked by mean norm:")
    for c in rank:
        print(f"  {CIFAR_CLASSES[c]:<10} μ={means[c]:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"ep{ckpt['epoch']}"

    # Plot 1: box/violin of per-class norms
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    order = np.argsort(means)
    data = [per_class[c] for c in order]
    positions = np.arange(10)
    axes[0].violinplot(data, positions=positions, showmeans=True, showmedians=False)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([CIFAR_CLASSES[c] for c in order], rotation=30, ha="right")
    axes[0].set_ylabel("L2 norm")
    axes[0].set_title("Feature norm distribution per class (sorted by μ)")
    axes[0].axhline(norms.mean(), color="gray", linestyle="--", alpha=0.5,
                    label=f"overall μ={norms.mean():.2f}")
    axes[0].legend(fontsize=8)

    # Plot 2: histogram overlay
    cmap = plt.cm.tab10
    for c in range(10):
        axes[1].hist(per_class[c], bins=30, alpha=0.35, histtype="step",
                     color=cmap(c), linewidth=1.5, label=CIFAR_CLASSES[c])
    axes[1].set_xlabel("L2 norm"); axes[1].set_ylabel("count")
    axes[1].set_title("Per-class norm histograms")
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle(f"Norm-by-class analysis (epoch {ckpt['epoch']}, head 256-d)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"norms_by_class_{tag}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")

    # Extra: how much of the norm variance is explained by class?
    between_class_var = np.var(means) * np.mean([len(per_class[c]) for c in range(10)])
    within_class_var = np.mean([per_class[c].var() for c in range(10)])
    total_var = norms.var()
    print(f"\n=== Variance decomposition of norm ===")
    print(f"Total variance:      {total_var:.4f}")
    print(f"Between-class var:   {np.var(means):.4f}")
    print(f"Within-class var:    {within_class_var:.4f}")
    print(f"R² of norm ~ class:  {np.var(means) / total_var:.4f}")


if __name__ == "__main__":
    main()

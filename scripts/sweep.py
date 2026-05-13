#!/usr/bin/env python
"""Run a sweep over named CovKL / baseline configurations on CIFAR-10.

Examples
--------
    # Run every config in configs.CONFIGS for 50 epochs
    python scripts/sweep.py --epochs 50

    # Run a subset
    python scripts/sweep.py --methods CovKL_v3_rn18 VICReg_rn18 --epochs 50

    # CPU smoke test
    python scripts/sweep.py --methods CovKL_v3_rn18 --epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow MPS fallbacks for ops that don't support it (e.g. some linalg).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Allow running directly from a clone without `pip install -e .`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from covkl.configs import CONFIGS, select
from covkl.train import select_device, train_and_eval


def _setup_style(style_path: str | None):
    """Use scienceplots + custom mplstyle if available, else fall back gracefully."""
    if not style_path:
        return
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", style_path])
    except Exception:
        try:
            plt.style.use(style_path)
        except Exception:
            pass  # use default matplotlib style


def _plot_knn_trajectories(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, h in results.items():
        ax.plot(h["epoch_knn"], h["knn_acc"], "o-", markersize=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("kNN accuracy")
    ax.set_title("CIFAR-10 kNN accuracy during pretraining")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final_bars(results: dict, out_path: str):
    names = list(results.keys())
    accs = [results[n]["knn_acc"][-1] for n in names]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(names))))
    bars = ax.barh(names, accs)
    ax.set_xlabel("Final kNN accuracy")
    ax.set_title("Final kNN accuracy by configuration")
    if accs:
        ax.set_xlim(min(accs) - 0.05, max(accs) + 0.02)
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_umap_and_norms(name: str, history: dict, out_dir: str):
    """Save UMAP embedding + norm histogram for one run. Skips if umap not installed."""
    try:
        import umap
    except ImportError:
        print("  [skip umap] install umap-learn to enable UMAP plots")
        return

    feats = history["final_features"]
    labels = history["labels"]
    print(f"  Fitting UMAP for {name}...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedded = reducer.fit_transform(feats)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(embedded[:, 0], embedded[:, 1], s=2, alpha=0.5,
                    c=labels, cmap="tab10")
    axes[0].set_title(f"{name} — UMAP (kNN={history['knn_acc'][-1]:.3f})")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    norms = np.linalg.norm(feats, axis=1)
    axes[1].hist(norms, bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_title(f"Feature norms: mean={norms.mean():.2f}  std={norms.std():.2f}")
    axes[1].set_xlabel("L2 norm")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"umap_{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Subset of {sorted(CONFIGS.keys())} (default: all).")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--log-dir", default="./runs/sweep")
    parser.add_argument("--ckpt-dir", default="./checkpoints/sweep")
    parser.add_argument("--results-dir", default="./results/sweep")
    parser.add_argument("--mplstyle", default=None,
                        help="Optional matplotlib style file (e.g. shreeyam.mplstyle).")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip per-config UMAP plots.")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Device: {device}")

    configs = select(args.methods)
    os.makedirs(args.results_dir, exist_ok=True)

    results = {}
    for name, config in configs.items():
        print(f"\n{'=' * 50}\nTraining: {name} | {config}\n{'=' * 50}")
        results[name] = train_and_eval(
            config,
            device=device,
            data_root=args.data_root,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            name=f"sweep_{name}",
            log_dir=args.log_dir,
            ckpt_dir=args.ckpt_dir,
        )

    _setup_style(args.mplstyle)
    _plot_knn_trajectories(results, os.path.join(args.results_dir, "sweep_knn.png"))
    _plot_final_bars(results, os.path.join(args.results_dir, "sweep_final.png"))

    for name, h in results.items():
        np.savez(
            os.path.join(args.results_dir, f"features_{name}.npz"),
            features=h["final_features"], labels=h["labels"],
        )
        if not args.skip_umap:
            _plot_umap_and_norms(name, h, args.results_dir)

    print(f"\nResults saved to {args.results_dir}/")
    print("\nFinal kNN accuracies:")
    for name in sorted(results.keys(), key=lambda n: results[n]["knn_acc"][-1], reverse=True):
        print(f"  {name}: {results[name]['knn_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()

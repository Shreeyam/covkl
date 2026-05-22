#!/usr/bin/env python
"""Run a sweep over named CovKL / baseline configurations.

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
import csv
import json
import os
import sys
from typing import Optional

# Allow MPS fallbacks for ops that don't support it (e.g. some linalg).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Allow running directly from a clone without `pip install -e .`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MPLCONFIGDIR = os.path.join(_REPO_ROOT, "results", ".mplconfig")
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from covkl.configs import CONFIGS, select
from covkl.train import select_device, train_and_eval


def _setup_style(style_path: Optional[str]):
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


def _plot_knn_trajectories(results: dict, out_path: str, dataset: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, h in results.items():
        ax.plot(h["epoch_knn"], h["knn_acc"], "o-", markersize=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("kNN accuracy")
    ax.set_title(f"{dataset} kNN accuracy during pretraining")
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


def _bool_arg(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = dict(config)
    overrides = {
        "arch": args.arch,
        "embed_dim": args.embed_dim,
        "projector_depth": args.projector_depth,
        "projector_hidden_dim": args.projector_hidden_dim,
        "projector_bn": args.projector_bn,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "rho": args.rho,
        "lam_align": args.lam_align,
        "lam_mu": args.lam_mu,
        "lam_covkl": args.lam_covkl,
        "lam_rkl": args.lam_rkl,
        "lam_skl": args.lam_skl,
        "lam_invar": args.lam_invar,
        "lam_var": args.lam_var,
        "lam_cov": args.lam_cov,
        "lam_bt": args.lam_bt,
    }
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def _run_name(dataset: str, method_name: str, config: dict, seed: int) -> str:
    depth = config.get("projector_depth", 3)
    width = config.get("projector_hidden_dim")
    if width is None:
        arch = config.get("arch", "smallconv")
        width = 1024 if arch in {"resnet18", "resnet18_imagenet", "resnet18-in"} else 512
    embed_dim = config.get("embed_dim", 256)
    return f"{dataset}_{method_name}_d{depth}_w{width}_z{embed_dim}_seed{seed}"


def _summary_for_run(run_name: str, dataset: str, method_name: str, config: dict,
                     seed: int, history: dict) -> dict:
    knn_acc = history.get("knn_acc", [])
    epoch_knn = history.get("epoch_knn", [])
    if knn_acc:
        best_idx = int(np.argmax(knn_acc))
        final_knn = float(knn_acc[-1])
        best_knn = float(knn_acc[best_idx])
        best_epoch = int(epoch_knn[best_idx])
    else:
        final_knn = None
        best_knn = None
        best_epoch = None
    return {
        "run_name": run_name,
        "dataset": dataset,
        "method_name": method_name,
        "method": config.get("method"),
        "config": config,
        "seed": seed,
        "final_knn": final_knn,
        "best_knn": best_knn,
        "best_epoch": best_epoch,
        "epochs_completed": history.get("epochs_completed"),
        "early_stopped": history.get("early_stopped", False),
        "wall_time_sec": history.get("wall_time_sec"),
    }


def _write_summaries(summaries: list[dict], results_dir: str):
    summary_dir = os.path.join(results_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    for summary in summaries:
        path = os.path.join(summary_dir, f"{summary['run_name']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    csv_path = os.path.join(results_dir, "sweep_summary.csv")
    fieldnames = [
        "run_name", "dataset", "method_name", "method", "seed", "final_knn",
        "best_knn", "best_epoch", "epochs_completed", "early_stopped",
        "wall_time_sec", "config",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            row["config"] = json.dumps(summary["config"], sort_keys=True)
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet100", "imagenet-100"])
    parser.add_argument("--image-size", type=int, default=None,
                        help="Input crop size. Defaults to 32 for CIFAR and 160 for ImageNet-100.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Single seed used when --seeds is not provided.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Run each selected method once per listed seed.")
    parser.add_argument("--knn-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=20)
    parser.add_argument("--early-stop-at", type=int, default=10,
                        help="Disable with --early-stop-at -1.")
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
    parser.add_argument("--arch", default=None,
                        choices=["resnet18", "resnet18_imagenet", "resnet18-in", "smallconv"])
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--projector-depth", type=int, default=None)
    parser.add_argument("--projector-hidden-dim", type=int, default=None)
    parser.add_argument("--projector-bn", type=_bool_arg, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--lam-align", type=float, default=None)
    parser.add_argument("--lam-mu", type=float, default=None)
    parser.add_argument("--lam-covkl", type=float, default=None)
    parser.add_argument("--lam-rkl", type=float, default=None)
    parser.add_argument("--lam-skl", type=float, default=None)
    parser.add_argument("--lam-invar", type=float, default=None)
    parser.add_argument("--lam-var", type=float, default=None)
    parser.add_argument("--lam-cov", type=float, default=None)
    parser.add_argument("--lam-bt", type=float, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Device: {device}")

    configs = select(args.methods)
    os.makedirs(args.results_dir, exist_ok=True)

    results = {}
    summaries = []
    seeds = args.seeds if args.seeds is not None else [args.seed]
    early_stop_at = None if args.early_stop_at < 0 else args.early_stop_at
    dataset = args.dataset.replace("-", "")
    for method_name, base_config in configs.items():
        for seed in seeds:
            config = _apply_overrides(base_config, args)
            run_name = _run_name(dataset, method_name, config, seed)
            print(f"\n{'=' * 50}\nTraining: {run_name} | {config}\n{'=' * 50}")
            history = train_and_eval(
                config,
                device=device,
                data_root=args.data_root,
                dataset=dataset,
                image_size=args.image_size,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                seed=seed,
                name=run_name,
                log_dir=args.log_dir,
                ckpt_dir=args.ckpt_dir,
                knn_every=args.knn_every,
                ckpt_every=args.ckpt_every,
                early_stop_at=early_stop_at,
            )
            results[run_name] = history
            summaries.append(_summary_for_run(run_name, dataset, method_name, config, seed, history))

    _setup_style(args.mplstyle)
    _plot_knn_trajectories(results, os.path.join(args.results_dir, "sweep_knn.png"), dataset)
    _plot_final_bars(results, os.path.join(args.results_dir, "sweep_final.png"))
    _write_summaries(summaries, args.results_dir)

    for name, h in results.items():
        np.savez(
            os.path.join(args.results_dir, f"features_{name}.npz"),
            features=h["final_features"], labels=h["labels"],
        )
        if not args.skip_umap:
            _plot_umap_and_norms(name, h, args.results_dir)

    print(f"\nResults saved to {args.results_dir}/")
    print("\nFinal kNN accuracies:")
    for summary in sorted(summaries, key=lambda s: s["final_knn"] or -1, reverse=True):
        print(f"  {summary['run_name']}: {summary['final_knn']:.4f}")


if __name__ == "__main__":
    main()

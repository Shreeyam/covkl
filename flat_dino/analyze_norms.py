"""Analyze feature geometry of a trained SimDINO checkpoint.

Checks how close the embedding distribution is to N(0, I) by comparing:
- per-dimension mean and variance
- off-diagonal correlations
- norm distribution vs chi distribution for N(0, I_d)
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
from scipy import stats
from torch.utils.data import DataLoader

from flat_dino.mini_experiment import ResNet18CIFAR, extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default="./results/mini")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-head", action="store_true",
                        help="Use projection head output (256-d) instead of backbone (512-d)")
    args = parser.parse_args()

    device = "mps" if (torch.backends.mps.is_available() and not args.cpu) else "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"Epoch {ckpt['epoch']}  config: {ckpt['config']}")

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

    # The training loss operates on projection head outputs, so evaluate those.
    # By default mini_experiment.extract_features uses forward_features (backbone).
    if args.use_head:
        @torch.no_grad()
        def get_feats():
            feats, labs = [], []
            for x, y in loader:
                feats.append(model(x.to(device)).cpu())
                labs.append(y)
            return torch.cat(feats).numpy(), torch.cat(labs).numpy()
        feats, labels = get_feats()
        space_name = "head (256-d)"
    else:
        feats, labels = extract_features(model, loader, device)
        space_name = "backbone (512-d)"

    N, D = feats.shape
    print(f"\nAnalyzing {N} features in {space_name}")

    # Per-dim stats
    mean_per_dim = feats.mean(axis=0)
    var_per_dim = feats.var(axis=0)
    std_per_dim = feats.std(axis=0)

    # Correlation matrix
    centered = feats - mean_per_dim
    corr = (centered.T @ centered) / N
    corr = corr / (std_per_dim[:, None] * std_per_dim[None, :] + 1e-8)
    off_diag_abs = np.abs(corr - np.eye(D))

    # Norms
    norms = np.linalg.norm(feats, axis=1)

    # Theoretical chi distribution for N(0, I_d): expected norm = sqrt(2) * Γ((d+1)/2) / Γ(d/2)
    from scipy.special import gammaln
    # log-domain to avoid overflow at large D
    log_mean = 0.5 * np.log(2) + gammaln((D + 1) / 2) - gammaln(D / 2)
    chi_mean = np.exp(log_mean)
    chi_var = D - chi_mean ** 2
    chi_std = np.sqrt(max(chi_var, 1e-8))

    print(f"\n=== Per-dimension statistics ===")
    print(f"Mean:       empirical μ={mean_per_dim.mean():.4f}  σ={mean_per_dim.std():.4f}  |  target μ=0")
    print(f"Variance:   empirical μ={var_per_dim.mean():.4f}  σ={var_per_dim.std():.4f}  |  target μ=1")
    print(f"Std:        empirical μ={std_per_dim.mean():.4f}  σ={std_per_dim.std():.4f}  |  target μ=1")
    print(f"\n=== Correlation ===")
    print(f"Mean |off-diag|:  {off_diag_abs[~np.eye(D, dtype=bool)].mean():.4f}")
    print(f"Max  |off-diag|:  {off_diag_abs[~np.eye(D, dtype=bool)].max():.4f}")
    print(f"\n=== Norm distribution ===")
    print(f"Empirical: μ={norms.mean():.3f}  σ={norms.std():.3f}")
    print(f"Chi(d={D}): μ={chi_mean:.3f}  σ={chi_std:.3f}")
    print(f"Ratio of mean norms: {norms.mean() / chi_mean:.3f}  (1.0 = perfect match)")

    # KS test against chi distribution
    # chi pdf with df=D, scale=1
    ks_stat, ks_p = stats.kstest(norms, stats.chi(df=D).cdf)
    print(f"KS test vs χ(d={D}): stat={ks_stat:.4f}, p={ks_p:.2e}")

    # === Plots ===
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"ep{ckpt['epoch']}_{space_name.split()[0]}"

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    # 1. Per-dim mean
    axes[0, 0].hist(mean_per_dim, bins=50, alpha=0.7, edgecolor="black",
                    color="steelblue")
    axes[0, 0].axvline(0, color="red", linestyle="--", label="target=0")
    axes[0, 0].set_title(f"Per-dim mean  (μ={mean_per_dim.mean():.3f})")
    axes[0, 0].set_xlabel("Mean"); axes[0, 0].legend(fontsize=8)

    # 2. Per-dim variance
    axes[0, 1].hist(var_per_dim, bins=50, alpha=0.7, edgecolor="black",
                    color="coral")
    axes[0, 1].axvline(1, color="red", linestyle="--", label="target=1")
    axes[0, 1].set_title(f"Per-dim variance  (μ={var_per_dim.mean():.3f})")
    axes[0, 1].set_xlabel("Variance"); axes[0, 1].legend(fontsize=8)

    # 3. Off-diagonal correlation magnitude
    mask = ~np.eye(D, dtype=bool)
    axes[0, 2].hist(off_diag_abs[mask], bins=50, alpha=0.7, edgecolor="black",
                    color="seagreen")
    axes[0, 2].set_title(f"|Off-diagonal correlation|  (μ={off_diag_abs[mask].mean():.3f})")
    axes[0, 2].set_xlabel("|corr|")

    # 4. Correlation matrix heatmap (subsample if large)
    if D > 256:
        idx = np.random.RandomState(0).choice(D, 256, replace=False)
        corr_plot = corr[np.ix_(idx, idx)]
    else:
        corr_plot = corr
    im = axes[1, 0].imshow(corr_plot, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1, 0].set_title("Correlation matrix")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.045)

    # 5. Norm distribution vs chi
    axes[1, 1].hist(norms, bins=50, alpha=0.7, edgecolor="black",
                    color="mediumpurple", density=True, label="empirical")
    xs = np.linspace(norms.min() * 0.9, norms.max() * 1.1, 500)
    axes[1, 1].plot(xs, stats.chi(df=D).pdf(xs), "r-", linewidth=2,
                    label=f"χ(d={D})")
    axes[1, 1].axvline(chi_mean, color="red", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(norms.mean(), color="black", linestyle="--", alpha=0.5)
    axes[1, 1].set_title(f"Norm: emp μ={norms.mean():.2f}, χ μ={chi_mean:.2f}")
    axes[1, 1].set_xlabel("L2 norm"); axes[1, 1].legend(fontsize=8)

    # 6. QQ plot: empirical norm vs chi quantiles
    emp_q = np.quantile(norms, np.linspace(0.01, 0.99, 99))
    chi_q = stats.chi(df=D).ppf(np.linspace(0.01, 0.99, 99))
    axes[1, 2].plot(chi_q, emp_q, "o", markersize=3, color="steelblue")
    lims = [min(emp_q.min(), chi_q.min()), max(emp_q.max(), chi_q.max())]
    axes[1, 2].plot(lims, lims, "r--", alpha=0.5, label="y=x")
    axes[1, 2].set_xlabel(f"χ(d={D}) quantiles")
    axes[1, 2].set_ylabel("Empirical norm quantiles")
    axes[1, 2].set_title("Q-Q plot")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(f"Gaussian-ness of {space_name} features  (epoch {ckpt['epoch']})",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"norms_analysis_{tag}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

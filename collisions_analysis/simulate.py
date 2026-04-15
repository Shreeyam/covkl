"""Monte Carlo verification of collision probability formulas for continuous Sidon sets."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_fn
from itertools import combinations
import os

import scienceplots
style_path = os.path.join(os.path.dirname(__file__), "..", "shreeyam.mplstyle")
plt.style.use(["science", style_path])

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Theoretical formulas from the paper
# ---------------------------------------------------------------------------

def vol_ball(d, eps):
    """Volume of a d-dimensional ball of radius eps."""
    return (np.pi ** (d / 2) / gamma_fn(d / 2 + 1)) * eps ** d


def f_Z_normal(d, sigma):
    """Density of Z at origin for Gaussian nodes: Z ~ N(0, 4σ²I)."""
    return 1.0 / (8 * np.pi * sigma ** 2) ** (d / 2)


def f_Z_ball(d):
    """Density of Z at origin for uniform-ball nodes: Z ~ N(0, 4/(d+2) I)."""
    return ((d + 2) / (8 * np.pi)) ** (d / 2)


def f_Z_sphere(d):
    """Density of Z at origin for uniform-sphere nodes: Z ~ N(0, 4/d I)."""
    return (d / (8 * np.pi)) ** (d / 2)


def lambda_expected(N, d, eps, f_Z_0):
    """Expected number of collisions: λ = (N⁴/8) * f_Z(0) * V_d(ε)."""
    M = N * (N - 1) / 2
    K = M * (M - 1) / 2
    return K * f_Z_0 * vol_ball(d, eps)


def P_c_theory(N, d, eps, f_Z_0):
    """Theoretical collision probability: P_c ≈ 1 - exp(-λ)."""
    lam = lambda_expected(N, d, eps, f_Z_0)
    return 1 - np.exp(-lam)


def capacity(d, eps, f_Z_0):
    """Expected capacity: N where P_c = 0.5."""
    numerator = 8 * np.log(2) * gamma_fn(d / 2 + 1)
    denominator = f_Z_0 * np.pi ** (d / 2) * eps ** d
    return (numerator / denominator) ** 0.25


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def sample_nodes(N, d, distribution, sigma=1.0):
    """Sample N nodes in d dimensions from the given distribution."""
    if distribution == "normal":
        return np.random.randn(N, d) * sigma
    elif distribution == "ball":
        # Uniform in unit ball: sample from Gaussian, normalize, scale by r^(1/d)
        X = np.random.randn(N, d)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        r = np.random.uniform(0, 1, (N, 1)) ** (1.0 / d)
        return X * r
    elif distribution == "sphere":
        # Uniform on unit sphere
        X = np.random.randn(N, d)
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def count_collisions(nodes, eps):
    """Count the number of collisions among pairwise difference vectors.

    A collision = two difference vectors within L2 distance eps.
    Returns (has_collision, n_collisions).
    """
    N = nodes.shape[0]
    # Compute all pairwise difference vectors
    diffs = []
    for i in range(N):
        for j in range(i + 1, N):
            diffs.append(nodes[i] - nodes[j])
    diffs = np.array(diffs)  # (M, d)
    M = len(diffs)

    # Check all pairs of difference vectors
    n_collisions = 0
    for i in range(M):
        for j in range(i + 1, M):
            if np.linalg.norm(diffs[i] - diffs[j]) <= eps:
                n_collisions += 1

    return n_collisions > 0, n_collisions


def count_collisions_fast(nodes, eps):
    """Vectorized collision counting - much faster for moderate N."""
    N = nodes.shape[0]
    # All pairwise differences: (M, d)
    idx = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])
    diffs = nodes[idx[:, 0]] - nodes[idx[:, 1]]
    M = len(diffs)

    # Compute pairwise distances between difference vectors
    # Do in chunks to avoid OOM
    has_collision = False
    n_collisions = 0
    chunk_size = 2000
    for i in range(0, M, chunk_size):
        chunk = diffs[i:i + chunk_size]
        # Distance from chunk to all diffs after index i
        for j in range(i, M, chunk_size):
            if j == i:
                # Within same chunk: only upper triangle
                d_mat = np.linalg.norm(chunk[:, None] - diffs[j:j + chunk_size][None, :], axis=-1)
                # Zero out lower triangle + diagonal
                mask = np.triu(np.ones(d_mat.shape, dtype=bool), k=1) if i == j else np.ones(d_mat.shape, dtype=bool)
            else:
                other = diffs[j:j + chunk_size]
                d_mat = np.linalg.norm(chunk[:, None] - other[None, :], axis=-1)
                mask = np.ones(d_mat.shape, dtype=bool)

            collisions_here = np.sum((d_mat <= eps) & mask)
            n_collisions += collisions_here
            if collisions_here > 0:
                has_collision = True

    return has_collision, n_collisions


def simulate_P_c(N, d, eps, distribution, n_trials=1000, sigma=1.0):
    """Estimate P_c by Monte Carlo. Returns (mean, ci_low, ci_high) with 95% CI."""
    hits = np.zeros(n_trials)
    for t in range(n_trials):
        nodes = sample_nodes(N, d, distribution, sigma=sigma)
        has_collision, _ = count_collisions_fast(nodes, eps)
        hits[t] = 1.0 if has_collision else 0.0
    p_hat = hits.mean()
    # 95% CI for a proportion (Wilson or normal approx)
    se = np.sqrt(p_hat * (1 - p_hat) / n_trials)
    return p_hat, max(0, p_hat - 1.96 * se), min(1, p_hat + 1.96 * se)


def simulate_collisions(N, d, eps, distribution, n_trials=1000, sigma=1.0):
    """Return mean collision count with 95% CI."""
    counts = np.zeros(n_trials)
    for t in range(n_trials):
        nodes = sample_nodes(N, d, distribution, sigma=sigma)
        _, nc = count_collisions_fast(nodes, eps)
        counts[t] = nc
    mean = counts.mean()
    se = counts.std() / np.sqrt(n_trials)
    return mean, mean - 1.96 * se, mean + 1.96 * se


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_Pc_vs_N():
    """Plot P_c vs N for each distribution across multiple dimensions."""
    eps = 0.5
    sigma = 1.0
    n_trials = 300
    d_values = [2, 3, 4, 5, 8]

    dist_configs = [
        ("normal", lambda d: f_Z_normal(d, sigma), r"Normal ($\sigma=1$)"),
        ("ball", f_Z_ball, "Uniform Ball"),
        ("sphere", f_Z_sphere, "Uniform Sphere"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for ax, (dist, fz0_fn, dist_label) in zip(axes, dist_configs):
        for d in d_values:
            fz0 = fz0_fn(d)
            # Pick N range based on where the transition is
            N_cap = capacity(d, eps, fz0)
            N_max = min(int(N_cap * 2.5), 40)
            N_values = np.arange(3, max(N_max, 8), max(1, (N_max - 3) // 10))
            N_theory = np.linspace(3, N_values[-1], 200)

            color = ax._get_lines.get_next_color()

            Pc_theory = [P_c_theory(n, d, eps, fz0) for n in N_theory]
            ax.plot(N_theory, Pc_theory, color=color, linewidth=1)

            means, ci_lo, ci_hi = [], [], []
            for N in N_values:
                print(f"  {dist_label} d={d}: N={N}")
                m, lo, hi = simulate_P_c(N, d, eps, dist, n_trials=n_trials, sigma=sigma)
                means.append(m); ci_lo.append(lo); ci_hi.append(hi)

            ax.plot(N_values, means, "o", color=color, markersize=2.5, label=f"$d={d}$")
            ax.fill_between(N_values, ci_lo, ci_hi, alpha=0.15, color=color)

        ax.set_xlabel("$N$")
        ax.set_ylabel("$P_c$")
        ax.set_title(dist_label)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.5)

    fig.suptitle(r"Collision Probability vs $N$ ($\varepsilon=0.5$, varying $d$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Pc_vs_N.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pc_vs_N.png")


def plot_Pc_vs_eps():
    """Plot P_c vs epsilon across multiple dimensions."""
    sigma = 1.0
    n_trials = 300
    d_values = [2, 3, 4, 5, 8]

    dist_configs = [
        ("normal", lambda d: f_Z_normal(d, sigma), r"Normal ($\sigma=1$)"),
        ("ball", f_Z_ball, "Uniform Ball"),
        ("sphere", f_Z_sphere, "Uniform Sphere"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for ax, (dist, fz0_fn, dist_label) in zip(axes, dist_configs):
        for d in d_values:
            fz0 = fz0_fn(d)
            # Pick a fixed N that's interesting for this d
            N = max(int(capacity(d, 0.5, fz0) * 0.8), 4)

            eps_values = np.linspace(0.1, 2.0, 12)
            eps_theory = np.linspace(0.1, 2.0, 200)

            color = ax._get_lines.get_next_color()

            Pc_theory = [P_c_theory(N, d, e, fz0) for e in eps_theory]
            ax.plot(eps_theory, Pc_theory, color=color, linewidth=1)

            means, ci_lo, ci_hi = [], [], []
            for e in eps_values:
                print(f"  {dist_label} d={d}: N={N}, eps={e:.2f}")
                m, lo, hi = simulate_P_c(N, d, e, dist, n_trials=n_trials, sigma=sigma)
                means.append(m); ci_lo.append(lo); ci_hi.append(hi)

            ax.plot(eps_values, means, "o", color=color, markersize=2.5,
                    label=f"$d={d}$, $N={N}$")
            ax.fill_between(eps_values, ci_lo, ci_hi, alpha=0.15, color=color)

        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("$P_c$")
        ax.set_title(dist_label)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=5)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.5)

    fig.suptitle(r"Collision Probability vs $\varepsilon$ (varying $d$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Pc_vs_eps.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pc_vs_eps.png")


def plot_capacity_vs_d():
    """Plot theoretical capacity vs dimension for each distribution."""
    eps = 0.5
    sigma = 1.0
    d_values = np.arange(2, 31)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    for dist_name, fz0_fn, label in [
        ("normal", lambda d: f_Z_normal(d, sigma), r"Normal ($\sigma=1$)"),
        ("ball", f_Z_ball, "Uniform Ball"),
        ("sphere", f_Z_sphere, "Uniform Sphere"),
    ]:
        caps = []
        for d in d_values:
            fz0 = fz0_fn(d) if callable(fz0_fn) else fz0_fn
            caps.append(capacity(d, eps, fz0))
        ax.semilogy(d_values, caps, marker="o", markersize=2, label=label)

    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Capacity $N$ (at $P_c = 0.5$)")
    ax.set_title(r"Theoretical Capacity vs Dimension ($\varepsilon=0.5$)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "capacity_vs_d.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved capacity_vs_d.png")


def plot_capacity_verification():
    """Verify capacity formula: for the predicted N, simulate P_c ≈ 0.5."""
    eps = 0.5
    sigma = 1.0
    n_trials = 300
    d_values = [2, 3, 4, 5, 6, 8, 10]

    configs = [
        ("normal", lambda d: f_Z_normal(d, sigma), "Normal"),
        ("ball", f_Z_ball, "Ball"),
        ("sphere", f_Z_sphere, "Sphere"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    for dist_name, fz0_fn, label in configs:
        means, ci_lo, ci_hi = [], [], []
        d_used = []
        for d in d_values:
            fz0 = fz0_fn(d)
            N_cap = capacity(d, eps, fz0)
            N_use = max(int(round(N_cap)), 3)
            if N_use > 35:
                print(f"  {label} d={d}: N={N_use} too large, skipping")
                continue
            d_used.append(d)
            print(f"  {label} d={d}: theoretical capacity N={N_cap:.1f}, using N={N_use}")
            m, lo, hi = simulate_P_c(N_use, d, eps, dist_name, n_trials=n_trials, sigma=sigma)
            means.append(m); ci_lo.append(lo); ci_hi.append(hi)
            print(f"    simulated P_c = {m:.3f} [{lo:.3f}, {hi:.3f}] (target: 0.5)")

        color = ax._get_lines.get_next_color()
        ax.plot(d_used, means, "o-", color=color, markersize=3, label=label)
        ax.fill_between(d_used, ci_lo, ci_hi, alpha=0.15, color=color)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8, label="Target $P_c=0.5$")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Simulated $P_c$ at theoretical capacity $N$")
    ax.set_title(r"Capacity Formula Verification ($\varepsilon=0.5$)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "capacity_verification.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved capacity_verification.png")


def plot_lambda_vs_collisions():
    """Compare expected λ with observed mean number of collisions across dimensions."""
    sigma = 1.0
    eps = 0.5
    n_trials = 300
    d_values = [2, 3, 4, 5]

    dist_configs = [
        ("normal", lambda d: f_Z_normal(d, sigma), "Normal"),
        ("ball", f_Z_ball, "Ball"),
        ("sphere", f_Z_sphere, "Sphere"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for ax, (dist, fz0_fn, dist_label) in zip(axes, dist_configs):
        for d in d_values:
            fz0 = fz0_fn(d)
            N_cap = capacity(d, eps, fz0)
            N_max = min(int(N_cap * 2), 20)
            N_values = np.arange(3, max(N_max, 6), max(1, (N_max - 3) // 6))

            color = ax._get_lines.get_next_color()

            N_theory = np.linspace(3, N_values[-1], 100)
            lam_theory = [lambda_expected(N, d, eps, fz0) for N in N_theory]
            ax.plot(N_theory, lam_theory, color=color, linewidth=1)

            means, ci_lo, ci_hi = [], [], []
            for N in N_values:
                print(f"  {dist_label} d={d}: N={N}")
                m, lo, hi = simulate_collisions(N, d, eps, dist, n_trials=n_trials, sigma=sigma)
                means.append(m); ci_lo.append(lo); ci_hi.append(hi)

            ax.plot(N_values, means, "o", color=color, markersize=2.5, label=f"$d={d}$")
            ax.fill_between(N_values, ci_lo, ci_hi, alpha=0.15, color=color)

        ax.set_xlabel("$N$")
        ax.set_ylabel(r"Expected collisions ($\lambda$)")
        ax.set_title(dist_label)
        ax.legend(fontsize=6)

    fig.suptitle(r"Expected vs Observed Collision Count ($\varepsilon=0.5$, varying $d$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lambda_vs_collisions.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lambda_vs_collisions.png")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 50)
    print("1. Capacity vs dimension (theoretical)")
    print("=" * 50)
    plot_capacity_vs_d()

    print("\n" + "=" * 50)
    print("2. P_c vs N (theory vs simulation)")
    print("=" * 50)
    plot_Pc_vs_N()

    print("\n" + "=" * 50)
    print("3. P_c vs epsilon (theory vs simulation)")
    print("=" * 50)
    plot_Pc_vs_eps()

    print("\n" + "=" * 50)
    print("4. λ vs observed collisions")
    print("=" * 50)
    plot_lambda_vs_collisions()

    print("\n" + "=" * 50)
    print("5. Capacity formula verification")
    print("=" * 50)
    plot_capacity_verification()

    print("\nAll plots saved to", OUT_DIR)

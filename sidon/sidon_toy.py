import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def make_pair_index(n: int, device=None):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return torch.tensor(pairs, dtype=torch.long, device=device)


def exact_metrics(E: torch.Tensor, eps: float):
    """Exact metrics on unordered disjoint pair differences."""
    device = E.device
    n = E.shape[0]
    pair_idx = make_pair_index(n, device)
    D = E[pair_idx[:, 1]] - E[pair_idx[:, 0]]
    dist = torch.cdist(D, D)

    a = pair_idx[:, None, 0]
    b = pair_idx[:, None, 1]
    c = pair_idx[None, :, 0]
    d = pair_idx[None, :, 1]

    # Upper triangle only, exclude shared endpoints.
    upper = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    disjoint = (a != c) & (a != d) & (b != c) & (b != d)
    mask = upper & disjoint

    masked = dist[mask]
    min_sep = masked.min().item()
    collisions = (masked < eps).sum().item()
    total = mask.sum().item()
    return {
        "num_pairs": D.shape[0],
        "num_pair_pairs": total,
        "min_sep": min_sep,
        "collision_count": collisions,
        "collision_rate": collisions / max(total, 1),
    }


def covariance(z: torch.Tensor):
    zc = z - z.mean(0, keepdim=True)
    return (zc.T @ zc) / max(z.shape[0] - 1, 1)


def whiten(z: torch.Tensor, eps: float = 1e-4):
    zc = z - z.mean(0, keepdim=True)
    cov = covariance(z)
    eye = torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov + eps * eye)
    W = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return zc @ W.T


def sample_disjoint_quadruples(n: int, q: int, device=None):
    out = []
    while len(out) < q:
        perms = torch.stack([torch.randperm(n, device=device)[:4] for _ in range(q - len(out))])
        out.append(perms)
    idx = torch.cat(out, dim=0)[:q]
    return idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]


def sidon_kernel_loss(z: torch.Tensor, num_quads: int, tau: float):
    """Smooth proxy for additive energy / near-collision density."""
    n = z.shape[0]
    i, j, k, l = sample_disjoint_quadruples(n, num_quads, z.device)
    q = z[i] - z[j] - z[k] + z[l]
    sq = (q * q).sum(dim=1)
    # log-mean-exp emphasizes hard near-collisions but stays smooth.
    return torch.logsumexp(-sq / (2.0 * tau * tau), dim=0) - math.log(num_quads)


def isotropy_loss(z: torch.Tensor):
    mu = z.mean(0)
    cov = covariance(z)
    eye = torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
    return (mu * mu).mean() + ((cov - eye) ** 2).mean()


def sliced_kurtosis_loss(z: torch.Tensor, num_dirs: int = 32):
    zc = z - z.mean(0, keepdim=True)
    dirs = torch.randn(num_dirs, z.shape[1], device=z.device, dtype=z.dtype)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    p = zc @ dirs.T
    second = (p ** 2).mean(dim=0)
    fourth = (p ** 4).mean(dim=0)
    excess = fourth / (second.clamp_min(1e-8) ** 2) - 3.0
    return (excess ** 2).mean()


@dataclass
class Config:
    n: int = 32
    d: int = 8
    steps: int = 2000
    lr: float = 2e-2
    tau_start: float = 1.0
    tau_end: float = 0.15
    num_quads: int = 4096
    lam_iso: float = 10.0
    lam_kurt: float = 0.5
    eps_eval: float = 0.35
    seed: int = 0
    device: str = "cpu"


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    E = torch.nn.Parameter(torch.randn(cfg.n, cfg.d, device=cfg.device))
    opt = torch.optim.Adam([E], lr=cfg.lr)

    print("Initial metrics:", exact_metrics(E.detach(), cfg.eps_eval))

    for step in range(1, cfg.steps + 1):
        t = (step - 1) / max(cfg.steps - 1, 1)
        tau = cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** t

        z_white = whiten(E)
        L_sid = sidon_kernel_loss(z_white, cfg.num_quads, tau)
        L_iso = isotropy_loss(E)
        L_kurt = sliced_kurtosis_loss(z_white)
        loss = L_sid + cfg.lam_iso * L_iso + cfg.lam_kurt * L_kurt

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(cfg.steps // 10, 1) == 0 or step == 1:
            m = exact_metrics(whiten(E.detach()), cfg.eps_eval)
            print(
                f"step={step:4d} tau={tau:.3f} loss={loss.item():.4f} "
                f"sid={L_sid.item():.4f} iso={L_iso.item():.4f} kurt={L_kurt.item():.4f} "
                f"min_sep={m['min_sep']:.4f} coll={m['collision_count']}"
            )

    final_E = whiten(E.detach())
    print("Final metrics:", exact_metrics(final_E, cfg.eps_eval))
    return final_E


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy optimization for Sidon-like embeddings.")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.15)
    parser.add_argument("--num-quads", type=int, default=4096)
    parser.add_argument("--lam-iso", type=float, default=10.0)
    parser.add_argument("--lam-kurt", type=float, default=0.5)
    parser.add_argument("--eps-eval", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    cfg = Config(
        n=args.n,
        d=args.d,
        steps=args.steps,
        lr=args.lr,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        num_quads=args.num_quads,
        lam_iso=args.lam_iso,
        lam_kurt=args.lam_kurt,
        eps_eval=args.eps_eval,
        seed=args.seed,
        device=args.device,
    )
    train(cfg)

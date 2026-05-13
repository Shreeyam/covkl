"""Self-supervised losses.

A loss function maps a pair of student embeddings (and optionally teacher
embeddings) to a scalar loss plus a dict of named components for logging.
Methods are registered in ``LOSSES`` and selected by name from ``config["method"]``.

Components dict always has the keys ``prediction``, ``variance``,
``decorrelation`` so downstream logging is uniform across methods.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

LossFn = Callable[..., Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
LOSSES: Dict[str, LossFn] = {}
NEEDS_TEACHER = {"simdino"}


def register(name: str):
    def deco(fn: LossFn) -> LossFn:
        LOSSES[name] = fn
        return fn
    return deco


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=x.device, dtype=x.dtype)


def _shrunk_cov(z: torch.Tensor, rho: float) -> torch.Tensor:
    """(1 - rho) * sample_cov(z) + rho * I."""
    N, D = z.shape
    c = z - z.mean(dim=0)
    C = (c.T @ c) / N
    I_D = torch.eye(D, device=z.device, dtype=z.dtype)
    return (1 - rho) * C + rho * I_D


# ---------------------------------------------------------------------------
# Covariance-KL Gaussian matching (this paper's family)
# ---------------------------------------------------------------------------

@register("covkl")
def covkl_loss(s1, s2, *, lam_align=25.0, lam_mu=1.0, lam_covkl=1.0, rho=1e-2, **_):
    """Forward Gaussian-KL on per-view covariance.

    Per-view:  ½ (tr(C) - logdet(C) - D) / D     (KL(N(μ,C) ‖ N(0,I)) on covariance)
    Mean:      mean(‖μ‖²)
    Align:     MSE(s1, s2)
    """
    loss_pred = F.mse_loss(s1, s2)

    def per_view(z):
        C_rho = _shrunk_cov(z, rho)
        _, logabsdet = torch.linalg.slogdet(C_rho)
        covkl = 0.5 * (torch.diagonal(C_rho).sum() - logabsdet - C_rho.shape[0]) / C_rho.shape[0]
        return covkl, (z.mean(dim=0) ** 2).mean()

    c1, mu1 = per_view(s1)
    c2, mu2 = per_view(s2)
    L_covkl = (c1 + c2) / 2
    L_mu = (mu1 + mu2) / 2
    loss = lam_align * loss_pred + lam_mu * L_mu + lam_covkl * L_covkl
    return loss, {"prediction": loss_pred, "variance": L_covkl, "decorrelation": L_mu}


@register("covkl_hinge")
def covkl_hinge_loss(s1, s2, *, lam_align=12.5, lam_mu=0.0, lam_covkl=1.0, rho=1e-2, **_):
    """CovKL⁺: hinge variant that only penalises eigenvalues of C below 1.

    L⁺(C) = (1/2) Σ_i (min(λ_i,1) - log min(λ_i,1) - 1)
    """
    loss_pred = F.mse_loss(s1, s2)

    def per_view(z):
        C_rho = _shrunk_cov(z, rho)
        eigvals = torch.linalg.eigvalsh(C_rho)
        clipped = torch.clamp(eigvals, max=1.0)
        covkl_plus = 0.5 * (clipped - torch.log(clipped) - 1.0).sum() / C_rho.shape[0]
        return covkl_plus, (z.mean(dim=0) ** 2).mean()

    c1, mu1 = per_view(s1)
    c2, mu2 = per_view(s2)
    L_covkl = (c1 + c2) / 2
    L_mu = (mu1 + mu2) / 2
    loss = lam_align * loss_pred + lam_mu * L_mu + lam_covkl * L_covkl
    return loss, {"prediction": loss_pred, "variance": L_covkl, "decorrelation": L_mu}


@register("covkl_hinge_split")
def covkl_hinge_split_loss(
    s1, s2, *, lam_align=25.0, lam_scale=25.0, lam_decor=1.0, rho=1e-2, **_,
):
    """VICReg-style decomposition of CovKL⁺:

    scale: CovKL-Diag⁺ on per-coordinate stds of shrunk cov (hinge at 1)
    decor: -½ logdet(R) / D where R is the correlation matrix of the shrunk cov
    """
    loss_pred = F.mse_loss(s1, s2)

    def per_view(z):
        C_rho = _shrunk_cov(z, rho)
        D = C_rho.shape[0]
        var = torch.diagonal(C_rho)
        sigma = torch.sqrt(var + 1e-8)
        f_diag = var - 2.0 * torch.log(sigma) - 1.0
        scale = 0.5 * torch.mean(torch.where(sigma < 1.0, f_diag, torch.zeros_like(f_diag)))
        inv_sigma = 1.0 / sigma
        R = C_rho * inv_sigma[:, None] * inv_sigma[None, :]
        _, logabsdet = torch.linalg.slogdet(R)
        decor = -0.5 * logabsdet / D
        return scale, decor

    s_sc1, s_dc1 = per_view(s1)
    s_sc2, s_dc2 = per_view(s2)
    L_scale = (s_sc1 + s_sc2) / 2
    L_decor = (s_dc1 + s_dc2) / 2
    loss = lam_align * loss_pred + lam_scale * L_scale + lam_decor * L_decor
    return loss, {"prediction": loss_pred, "variance": L_scale, "decorrelation": L_decor}


@register("symkl")
def symkl_loss(s1, s2, *, lam_align=25.0, lam_skl=0.01, rho=1e-2, **_):
    """Symmetric KL (Jeffreys) covariance matching: ½(tr C + tr C⁻¹ - 2D)/D."""
    loss_pred = F.mse_loss(s1, s2)

    def per_view(z):
        C_rho = _shrunk_cov(z, rho)
        D = C_rho.shape[0]
        tr_C = torch.diagonal(C_rho).sum()
        tr_Cinv = torch.diagonal(torch.linalg.inv(C_rho)).sum()
        return 0.5 * (tr_C + tr_Cinv - 2 * D) / D

    L_skl = (per_view(s1) + per_view(s2)) / 2
    loss = lam_align * loss_pred + lam_skl * L_skl
    return loss, {"prediction": loss_pred, "variance": L_skl, "decorrelation": _zero_like(loss_pred)}


@register("revkl")
def revkl_loss(s1, s2, *, lam_align=25.0, lam_rkl=0.01, rho=1e-2, **_):
    """Reverse Gaussian-KL (KL(q‖p), q=N(0,I)): ½(tr C⁻¹ + logdet C - D)/D.

    Inverse-eigenvalue barrier on the collapse side; only log growth on the
    expansion side (the opposite asymmetry to forward CovKL).
    """
    loss_pred = F.mse_loss(s1, s2)

    def per_view(z):
        C_rho = _shrunk_cov(z, rho)
        D = C_rho.shape[0]
        tr_Cinv = torch.diagonal(torch.linalg.inv(C_rho)).sum()
        _, logabsdet = torch.linalg.slogdet(C_rho)
        return 0.5 * (tr_Cinv + logabsdet - D) / D

    L_rkl = (per_view(s1) + per_view(s2)) / 2
    loss = lam_align * loss_pred + lam_rkl * L_rkl
    return loss, {"prediction": loss_pred, "variance": L_rkl, "decorrelation": _zero_like(loss_pred)}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@register("vicreg")
def vicreg_loss(s1, s2, *, lam_invar=25.0, lam_var=25.0, lam_cov=1.0, **_):
    """VICReg (Bardes et al. 2022): invariance + variance hinge + covariance off-diag."""
    loss_pred = F.mse_loss(s1, s2)

    def per_view(z):
        N, D = z.shape
        z_c = z - z.mean(dim=0)
        std = torch.sqrt(z_c.var(dim=0, unbiased=False) + 1e-4)
        v_hinge = torch.mean(F.relu(1.0 - std))
        cov = (z_c.T @ z_c) / (N - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        return v_hinge, (off ** 2).sum() / D

    v1_h, c1 = per_view(s1)
    v2_h, c2 = per_view(s2)
    L_var = (v1_h + v2_h) / 2
    L_cov = (c1 + c2) / 2
    loss = lam_invar * loss_pred + lam_var * L_var + lam_cov * L_cov
    return loss, {"prediction": loss_pred, "variance": L_var, "decorrelation": L_cov}


@register("barlow")
def barlow_loss(s1, s2, *, lam_bt=5e-3, **_):
    """Barlow Twins (Zbontar et al. 2021): cross-correlation diagonal + off-diagonal."""
    N, D = s1.shape

    def bn(z):
        return (z - z.mean(dim=0)) / (z.std(dim=0, unbiased=False) + 1e-4)

    z1n, z2n = bn(s1), bn(s2)
    C = (z1n.T @ z2n) / N
    on_diag = ((torch.diagonal(C) - 1.0) ** 2).sum()
    off = C - torch.diag(torch.diagonal(C))
    off_diag = (off ** 2).sum()
    loss = on_diag + lam_bt * off_diag
    return loss, {"prediction": on_diag, "variance": _zero_like(on_diag), "decorrelation": off_diag}


@register("simdino")
def simdino_loss(
    s1, s2, *, t1, t2, cr_eps=0.5, gamma=1.0, w_sigreg=5.0, **_,
):
    """SimDINO-style: cross-view MSE + correlation-rate maximisation + SigReg.

    Requires a teacher (``NEEDS_TEACHER``); ``t1``/``t2`` are detached teacher
    embeddings of the two views.
    """
    loss_pred = 0.5 * (F.mse_loss(s1, t2.detach()) + F.mse_loss(s2, t1.detach())) / 2

    feats = torch.cat([s1, s2], dim=0)
    N, D = feats.shape
    centered = feats - feats.mean(dim=0)
    std_per_dim = centered.std(dim=0, unbiased=False) + 1e-4
    normed = centered / std_per_dim
    corr = (normed.T @ normed) / N
    I_D = torch.eye(D, device=feats.device, dtype=feats.dtype)
    _, logabsdet = torch.linalg.slogdet(I_D + (D / cr_eps ** 2) * corr)
    R_corr = 0.5 * logabsdet

    loss_sigreg = torch.mean((std_per_dim - 1.0) ** 2)
    loss = loss_pred + w_sigreg * loss_sigreg - gamma * R_corr
    return loss, {"prediction": loss_pred, "variance": loss_sigreg, "decorrelation": -R_corr}


def compute_loss(method: str, s1, s2, *, t1=None, t2=None, **kwargs):
    """Dispatch to a registered loss by name."""
    if method not in LOSSES:
        raise ValueError(f"unknown method: {method!r}. Available: {sorted(LOSSES)}")
    fn = LOSSES[method]
    if method in NEEDS_TEACHER:
        return fn(s1, s2, t1=t1, t2=t2, **kwargs)
    return fn(s1, s2, **kwargs)

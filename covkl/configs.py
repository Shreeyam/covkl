"""Named experiment configurations used by ``scripts/sweep.py``.

Each entry maps a short name (used as the sweep run id and TensorBoard
sub-directory) to a config dict. The config dict is passed straight to
``covkl.train.train_and_eval``.
"""

from __future__ import annotations


def _sim(**overrides):
    base = dict(method="simdino", lr=1e-3, ema_decay=0.99,
                cr_eps=0.5, gamma=1.0, w_sigreg=5.0)
    return {**base, **overrides}


CONFIGS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # CovKL family (this paper)
    # ------------------------------------------------------------------
    "CovKL_rn18": dict(method="covkl", lr=1e-3, arch="resnet18",
                       lam_align=25.0, lam_mu=1.0, lam_covkl=1.0, rho=1e-2),
    "CovKL_v2_rn18": dict(method="covkl", lr=1e-3, arch="resnet18",
                          lam_align=50.0, lam_mu=0.0, lam_covkl=1.0, rho=1e-2),
    "CovKL_v3_rn18": dict(method="covkl", lr=1e-3, arch="resnet18",
                          lam_align=12.5, lam_mu=0.0, lam_covkl=1.0, rho=1e-2),
    "CovKL_hinge_rn18": dict(method="covkl_hinge", lr=1e-3, arch="resnet18",
                             lam_align=12.5, lam_mu=0.0, lam_covkl=1.0, rho=1e-2),
    "CovKL_hinge_vic_rn18": dict(method="covkl_hinge_split", lr=1e-3, arch="resnet18",
                                 lam_align=25.0, lam_scale=25.0, lam_decor=1.0, rho=1e-2),

    # Symmetric KL (Jeffreys)
    "SymKL_rn18": dict(method="symkl", lr=1e-3, arch="resnet18",
                       lam_align=25.0, lam_skl=0.5, rho=1e-2),
    "SymKL_v2_rn18": dict(method="symkl", lr=1e-3, arch="resnet18",
                          lam_align=25.0, lam_skl=0.02, rho=1e-2),
    "SymKL_v3_rn18": dict(method="symkl", lr=1e-3, arch="resnet18",
                          lam_align=25.0, lam_skl=0.01, rho=1e-2),
    "SymKL_v4_rn18": dict(method="symkl", lr=1e-3, arch="resnet18",
                          lam_align=12.5, lam_skl=0.01, rho=1e-2),
    "SymKL_v5_rn18": dict(method="symkl", lr=1e-3, arch="resnet18",
                          lam_align=25.0, lam_skl=0.005, rho=1e-2),
    "SymKL_v6_rn18": dict(method="symkl", lr=1e-3, arch="resnet18",
                          lam_align=25.0, lam_skl=0.01, rho=5e-2),
    "SymKL_v7_rn18": dict(method="symkl", lr=1e-3, arch="resnet18",
                          lam_align=12.5, lam_skl=0.005, rho=1e-2),

    # Reverse KL
    "RevKL_rn18": dict(method="revkl", lr=1e-3, arch="resnet18",
                       lam_align=25.0, lam_rkl=0.01, rho=1e-2),
    "RevKL_v2_rn18": dict(method="revkl", lr=1e-3, arch="resnet18",
                          lam_align=12.5, lam_rkl=0.005, rho=1e-2),
    "RevKL_v3_rn18": dict(method="revkl", lr=1e-3, arch="resnet18",
                          lam_align=12.5, lam_rkl=0.01, rho=1e-2),
    "RevKL_v4_rn18": dict(method="revkl", lr=1e-3, arch="resnet18",
                          lam_align=25.0, lam_rkl=0.02, rho=1e-2),
    "RevKL_v5_rn18": dict(method="revkl", lr=1e-3, arch="resnet18",
                          lam_align=50.0, lam_rkl=0.01, rho=1e-2),
    "RevKL_v6_rn18": dict(method="revkl", lr=1e-3, arch="resnet18",
                          lam_align=25.0, lam_rkl=0.005, rho=1e-2),
    "RevKL_v7_rn18": dict(method="revkl", lr=1e-3, arch="resnet18",
                          lam_align=50.0, lam_rkl=0.005, rho=1e-2),

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    "SimDINO_rn18_long": _sim(gamma=1e-3, w_sigreg=1.0, arch="resnet18"),
    "VICReg_rn18": dict(method="vicreg", lr=1e-3, arch="resnet18",
                        lam_invar=25.0, lam_var=25.0, lam_cov=1.0),
    "Barlow_rn18": dict(method="barlow", lr=1e-3, arch="resnet18",
                        lam_bt=5e-3),
}


def select(names: list[str] | None) -> dict[str, dict]:
    """Return CONFIGS filtered to ``names`` (or all if ``names`` is None)."""
    if not names:
        return dict(CONFIGS)
    missing = set(names) - set(CONFIGS)
    if missing:
        raise ValueError(f"unknown configs: {sorted(missing)}. "
                         f"Available: {sorted(CONFIGS)}")
    return {k: CONFIGS[k] for k in names}

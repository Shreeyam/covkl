# covkl

Gaussian-KL covariance matching for self-supervised learning on CIFAR-10.

This repository accompanies experiments on **CovKL** — a self-supervised
loss that regularises the per-view feature covariance by minimising the
Gaussian KL divergence between the empirical covariance and the identity:

$$\mathcal{L}_{\text{CovKL}}(C) = \tfrac{1}{2D}\bigl(\operatorname{tr}(C) - \log\det(C) - D\bigr).$$

Per-eigenvalue this is `½ (λ - log λ - 1)`, an asymmetric barrier that
prevents collapse logarithmically and penalises over-expansion linearly.
Alongside CovKL the package implements its hinge, split, symmetric- and
reverse-KL variants, plus VICReg, Barlow Twins, and a SimDINO-style
baseline.

## Install

```bash
git clone <repo-url> && cd covkl
pip install -e .[plots]
```

The `[plots]` extra adds `umap-learn` and `scienceplots` used by the
sweep visualisations. Core training runs without them.

## Quick start

A two-epoch CPU smoke test on a single config:

```bash
python scripts/sweep.py --methods CovKL_v3_rn18 --epochs 2 --device cpu --skip-umap
```

Full sweep used in the paper (ResNet-18, CIFAR-10):

```bash
python scripts/sweep.py --epochs 50 --batch-size 512
```

CIFAR-10 is downloaded to `./data` on first run. Per-step training
losses and per-epoch kNN accuracy are streamed to TensorBoard under
`./runs/sweep/`; final feature arrays, UMAP plots, and the summary
bar chart go to `./results/sweep/`.

## Methods

| name (`config["method"]`)     | description                                                       |
| ----------------------------- | ----------------------------------------------------------------- |
| `covkl`                       | Forward Gaussian-KL on per-view covariance (this work).           |
| `covkl_hinge`                 | CovKL⁺: only eigenvalues of `C` below 1 are penalised.            |
| `covkl_hinge_split`           | VICReg-style split of CovKL⁺ into scale + correlation terms.      |
| `symkl`                       | Symmetric KL (Jeffreys): `½ (tr C + tr C⁻¹ - 2D) / D`.            |
| `revkl`                       | Reverse KL: `½ (tr C⁻¹ + log det C - D) / D`.                     |
| `vicreg`                      | VICReg (Bardes et al. 2022).                                      |
| `barlow`                      | Barlow Twins (Zbontar et al. 2021).                               |
| `simdino`                     | Correlation-rate + SigReg with an EMA teacher (SimDINO-style).    |

All methods share the same backbone (`resnet18` adapted for 32×32, or
`smallconv`), the same two-view CIFAR-10 augmentation pipeline
(`covkl.data.MultiCropCIFAR`), and the same AdamW + cosine schedule.

## Layout

```
covkl/
  __init__.py
  models.py     # ResNet18CIFAR, SmallConvNet, build_encoder
  data.py       # MultiCropCIFAR, build_loaders
  losses.py     # registry of SSL losses (compute_loss / LOSSES)
  train.py      # train_and_eval, select_device
  eval.py       # kNN evaluation and feature extraction
  configs.py    # named experiment configs used by the sweep
scripts/
  sweep.py      # CLI entry point
```

## Adding a new loss

```python
# covkl/losses.py
@register("my_loss")
def my_loss(s1, s2, *, lam_align=25.0, my_param=1.0, **_):
    pred = F.mse_loss(s1, s2)
    reg = ...  # your regulariser on s1 / s2 or their covariances
    loss = lam_align * pred + my_param * reg
    return loss, {"prediction": pred, "variance": reg,
                  "decorrelation": torch.zeros_like(pred)}
```

Then add a named config to `covkl/configs.py` referencing `method="my_loss"`
and run it through `scripts/sweep.py --methods my_config_name`.

If the loss needs an EMA teacher, add the method name to
`covkl.losses.NEEDS_TEACHER` and accept `t1`, `t2` kwargs.

## License

MIT — see [LICENSE](LICENSE).

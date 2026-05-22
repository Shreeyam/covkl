# CovKL Experimentation Plan

This document is intended as an execution plan for running the next round of
experiments on a larger GPU machine. The goal is to turn the current paper from
a promising CIFAR-10 prototype into a defensible experimental story.

## Current Status

The current repo already has CIFAR-10 SSL training, kNN evaluation, TensorBoard
logging, and sweep plotting. Existing 200-epoch CIFAR-10 runs show that the KL
family is competitive with flat-space baselines:

| Method | Final kNN | Best kNN |
| --- | ---: | ---: |
| RevKL_rn18 | 83.95 | 84.11 |
| CovKL_v3_rn18 | 83.80 | 83.88 |
| CovKL_rn18 | 83.59 | 83.72 |
| SymKL_v4_rn18 | 83.63 | 83.68 |
| VICReg_rn18 | 83.50 | 83.57 |
| Barlow_rn18 | 83.31 | 83.44 |

These numbers are promising but not yet paper-grade. The margins are small, the
best hyperparameters are not uniformly tuned, and the projector capacity is
currently fixed in code.

## Main Questions

1. Which Gaussian-KL direction is best under a matched SSL recipe?
2. Does the answer survive hyperparameter tuning?
3. Is the apparent gain caused by the objective, or by projector capacity?
4. Do the results transfer from CIFAR-10 to CIFAR-100 and ImageNet-100?
5. Do the covariance spectra behave as predicted by the paper's spectral
   analysis?

## Datasets

Use three datasets, in this order:

| Dataset | Role | Notes |
| --- | --- | --- |
| CIFAR-10 | Main tuning dataset | Fast enough for broad sweeps. Use for loss weights, shrinkage, LR, and projector sweeps. |
| CIFAR-100 | Robustness check | Same resolution/pipeline, harder labels. Run reduced sweeps and final repeats. |
| ImageNet-100 | Scale check | ImageNet-like transfer dataset with 100 classes. |

## Repo Options Available for Sweeps

### 1. Dataset selection

`scripts/sweep.py` supports:

- `cifar10`
- `cifar100`
- `imagenet100`

Example:

```bash
python scripts/sweep.py \
  --dataset imagenet100 \
  --data-root ./data \
  --arch resnet18_imagenet \
  --image-size 160 \
  --methods RevKL_rn18 VICReg_rn18 Barlow_rn18 \
  --device cuda
```

Evaluation must set `num_classes` correctly for kNN:

| Dataset | Classes |
| --- | ---: |
| CIFAR-10 | 10 |
| CIFAR-100 | 100 |
| ImageNet-100 | 100 |

### 2. Projector configuration

The projector can be controlled from the CLI:

- `projector_depth`
- `projector_hidden_dim`
- `embed_dim`
- `projector_bn`

The default reproduces the current ResNet-18 architecture:

```bash
--projector-depth 3 --projector-hidden-dim 1024 --embed-dim 256
```

### 3. Reproducibility

The sweep entrypoint supports explicit seeds:

- Python random
- NumPy
- Torch CPU
- CUDA
- DataLoader workers

Run names include dataset, seed, method, and projector config:

```text
{dataset}_{method}_d{projector_depth}_w{projector_hidden_dim}_z{embed_dim}_seed{seed}
```

### 4. Result summaries

Each run writes a machine-readable summary file, not just TensorBoard:

```json
{
  "run_name": "...",
  "dataset": "cifar10",
  "method": "revkl",
  "config": {},
  "seed": 0,
  "final_knn": 0.8395,
  "best_knn": 0.8411,
  "best_epoch": 190,
  "epochs_completed": 200,
  "early_stopped": false,
  "wall_time_sec": 0
}
```

The repo writes one JSON per run under `results/sweep/summaries/` and one
aggregate CSV at `results/sweep/sweep_summary.csv`.

## Stage 0: Smoke Tests

Purpose: verify data loading, projectors, seeds, summaries, and GPU execution.

Run on CIFAR-10 for 1-2 epochs:

```bash
python scripts/sweep.py \
  --dataset cifar10 \
  --methods CovKL_v3_rn18 RevKL_rn18 SymKL_v4_rn18 VICReg_rn18 Barlow_rn18 \
  --epochs 2 \
  --batch-size 256 \
  --device cuda \
  --skip-umap
```

Repeat for CIFAR-100 and ImageNet-100 with one method each.

Success criteria:

- no NaNs
- kNN eval runs with the right class count
- checkpoints and JSON summaries are written
- run names are unique and traceable

## Stage 1: CIFAR-10 Hyperparameter Sweep

Purpose: find strong, matched recipes for each objective under the current
default projector.

Fixed:

- dataset: CIFAR-10
- backbone: ResNet-18 CIFAR stem
- projector: depth 3, hidden 1024, output 256
- epochs: 200
- batch size: 256 or 512, but keep fixed across all methods
- optimizer: AdamW
- schedule: cosine
- evaluation: kNN every 10 or 20 epochs

Sweep these methods:

- `covkl`
- `revkl`
- `symkl`
- `vicreg`
- `barlow`

Suggested KL-family grid:

| Method | Sweep |
| --- | --- |
| CovKL | `lam_align in {12.5, 25, 50}`, `lam_covkl in {0.5, 1, 2}`, `rho in {1e-3, 1e-2, 5e-2}` |
| RevKL | `lam_align in {12.5, 25, 50}`, `lam_rkl in {0.002, 0.005, 0.01, 0.02}`, `rho in {1e-3, 1e-2, 5e-2}` |
| SymKL | `lam_align in {12.5, 25, 50}`, `lam_skl in {0.002, 0.005, 0.01, 0.02}`, `rho in {1e-3, 1e-2, 5e-2}` |

These are exposed directly as CLI overrides: `--lam-align`, `--lam-covkl`,
`--lam-rkl`, `--lam-skl`, and `--rho`. Keep `--lam-mu 0` for this stage so
the main sweep isolates covariance geometry.

Baselines should use fixed paper-style defaults rather than being tuned as part
of the main claim:

| Method | Fixed recipe |
| --- | --- |
| VICReg | `lam_invar=25`, `lam_var=25`, `lam_cov=1` |
| Barlow Twins | `lam_bt=5e-3` |

Run Stage 1 KL-family configs with seed `0`. Keep top 3 KL configs per method by
best kNN. Carry the fixed VICReg and Barlow recipes as reference baselines.

## Stage 2: Mean-Term Ablation

Purpose: test whether first-moment control helps after the covariance objective
has already been tuned.

Use only the best Stage 1 config for each KL-family method:

- best `CovKL`
- best `RevKL`
- best `SymKL`

Sweep a shared isotropic mean penalty:

| Parameter | Values |
| --- | --- |
| `lam_mu` | `0, 0.1, 1.0` |

Use the same mean penalty for all KL directions:

```text
L_mu = mean(mu^2)
```

Do not use direction-specific exact Gaussian mean terms in the main experiment.
The exact reverse/symmetric terms introduce `C^{-1}` into the mean penalty and
would confound this ablation with an additional conditioning/stability variable.

CLI flag:

```bash
--lam-mu 0.1
```

Selection rule: if nonzero `lam_mu` clearly improves a KL method, carry the best
`lam_mu` into later stages for that method. Otherwise keep `lam_mu=0`.

## Stage 3: Projector Capacity Sweep

Purpose: test whether KL-family gains persist under matched projector capacity.

Use the best Stage 1/2 config for each method. Sweep:

| Parameter | Values |
| --- | --- |
| `projector_depth` | `1, 2, 3, 4` |
| `projector_hidden_dim` | `64, 128, 256, 512, 1024, 2048` |
| `embed_dim` | `128, 256, 512` |

Run the full Cartesian product. This is 72 projector settings per method:

```text
4 depths x 6 widths x 3 output dimensions
```

Methods to include:

- best `CovKL`
- best `RevKL`
- best `SymKL`
- fixed-recipe `VICReg`
- fixed-recipe `Barlow`

Primary comparison:

```text
best kNN versus projector parameter count
```

The paper needs this because covariance losses act directly in projector space.
If KL methods only win at one head size, the result is less convincing.

## Stage 4: Seed Repeats on CIFAR-10

Purpose: estimate variance for the final CIFAR-10 table.

Take the best projector and loss config per method from Stages 1-3. Run:

```text
seeds = 0, 1, 2
```

Prefer 5 seeds if the A100 is idle, but 3 seeds is the minimum useful bar.

Report:

- final kNN mean/std
- best kNN mean/std
- best epoch mean/std
- wall-clock time per epoch

## Stage 5: CIFAR-100 Transfer

Purpose: test whether the tuned recipes survive a harder same-resolution
dataset.

Use the selected CIFAR-10 recipes. Do not retune broadly at first.

Run:

- best `CovKL`
- best `RevKL`
- best `SymKL`
- fixed-recipe `VICReg`
- fixed-recipe `Barlow`

Use seeds:

```text
0, 1, 2
```

If all KL-family methods degrade badly, run a small CIFAR-100-only LR and
regularization sweep. Otherwise keep the CIFAR-10-selected recipes for a cleaner
transfer story.

## Stage 6: ImageNet-100 Scale Check

Purpose: show the objective is not CIFAR-specific.

Use fewer methods:

- best KL-family method overall
- best forward KL if different
- VICReg
- Barlow

Recommended setup:

- backbone: standard-stem ResNet-18 via `--arch resnet18_imagenet`
- image size: 128 or 160 if compute allows
- epochs: 100-200 depending on runtime
- batch size: maximize within memory, but keep fixed across methods
- seeds: `0, 1, 2` if feasible; otherwise seed `0` plus a note that this is a scale check

## Stage 7: Spectral Diagnostics

Purpose: directly test the paper's spectral claims.

For each final method checkpoint, compute covariance eigenspectra on frozen
backbone features and projector features:

- top eigenvalue
- bottom eigenvalue
- condition number
- effective rank
- spectral entropy
- fraction of eigenvalues below `0.1`, `0.5`, and `1.0`
- fraction above `1.0`, `2.0`, and `5.0`

Save per-epoch spectra for at least:

- CovKL
- RevKL
- SymKL
- VICReg
- Barlow

Expected qualitative patterns:

- Forward KL should suppress over-expanded directions more strongly.
- Reverse KL should lift small eigenvalues more aggressively.
- Symmetric KL should be conservative on both tails.
- VICReg may keep coordinate variance healthy while still allowing spectral
  imbalance.
- Barlow may behave differently because it targets cross-correlation rather than
  per-view covariance.

## Paper Tables and Figures

### Main CIFAR table

| Method | CIFAR-10 final | CIFAR-10 best | CIFAR-100 final | CIFAR-100 best | Params | Time/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

Use mean ± std over seeds.

### Projector sweep figure

Plot best kNN against:

- projector depth
- projector width
- output dimension
- projector parameter count

Include at least KL winner, VICReg, and Barlow.

### Spectral figure

For each method, plot:

- eigenvalue distribution at final epoch
- effective rank over training
- min/max eigenvalue over training

### Hyperparameter appendix

Include a compact table of the selected hyperparameters and the sweep ranges.

## Execution Priority

If compute is limited, use this order:

1. Validate dataset/projector/seed/result-summary support.
2. Smoke test all datasets.
3. CIFAR-10 Stage 1 hyperparameter sweep.
4. CIFAR-10 Stage 2 mean-term ablation on best KL configs.
5. CIFAR-10 Stage 3 projector sweep for top methods.
6. CIFAR-10 3-seed repeats.
7. CIFAR-100 3-seed repeats.
8. ImageNet-100 scale check.
9. Spectral diagnostics and paper figures.

## Minimal Final Experimental Claim

The minimum publishable experimental claim should look like:

> Under a matched ResNet-18 Siamese SSL recipe, Gaussian-KL covariance matching
> is competitive with VICReg and Barlow Twins on CIFAR-10 and CIFAR-100. The KL
> direction controls the learned covariance spectrum as predicted: reverse KL
> more strongly protects collapsed eigendirections, while forward KL more
> strongly limits over-expanded eigendirections. These effects persist across
> projector capacity sweeps.

Avoid claiming a large accuracy win unless the seed repeats support it.

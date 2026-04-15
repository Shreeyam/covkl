"""Test whether the norm-as-confidence phenomenon we observed in our flat-space
SimDINO holds for pretrained DINOv2.

DINOv2 is a spherical-SSL model (in contrast to our flat-Gaussian setup). If the
phenomenon is specific to flat-space, DINOv2 norms should NOT shrink cleanly
with noise. If it's more general, DINOv2 should show the same pattern.
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


CIFAR_CLASSES = ["airplane", "auto", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def dinov2_cls_backbone(model, x):
    """Return the (unnormalized) CLS token from the ViT trunk."""
    out = model.forward_features(x)
    return out["x_norm_clstoken"]  # (B, D)


@torch.no_grad()
def dinov2_patch_mean(model, x):
    """Return mean-pooled patch tokens (alternative representation)."""
    out = model.forward_features(x)
    return out["x_norm_patchtokens"].mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--n-alphas", type=int, default=11)
    parser.add_argument("--out-dir", default="./results/mini")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "mps" if (torch.backends.mps.is_available() and not args.cpu) else "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"

    print(f"Loading {args.model}...")
    model = torch.hub.load("facebookresearch/dinov2", args.model, pretrained=True)
    model = model.to(device).eval()
    D = model.embed_dim
    print(f"Model embed dim = {D}")

    # DINOv2 expects 224x224 (or multiples of 14). Upsample CIFAR from 32x32.
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    # Load CIFAR images as raw tensors then upsample
    to_tensor = T.ToTensor()
    raw_ds = torchvision.datasets.CIFAR10("./data", train=False, download=False)

    class_imgs = {c: [] for c in range(10)}
    for i in range(len(raw_ds)):
        img, lbl = raw_ds[i]
        if len(class_imgs[lbl]) < args.n_per_class:
            class_imgs[lbl].append(to_tensor(img))
        if all(len(v) >= args.n_per_class for v in class_imgs.values()):
            break

    all_imgs = torch.stack([img for c in range(10) for img in class_imgs[c]])
    all_labels = np.array([c for c in range(10) for _ in range(args.n_per_class)])
    N = len(all_imgs)
    print(f"Loaded {N} CIFAR images (32x32) — will upsample to 224x224")

    torch.manual_seed(0)
    noise = torch.rand_like(all_imgs)
    alphas = np.linspace(0.0, 1.0, args.n_alphas)

    def prep(x_32):
        """Mix with noise is done before this; this just upsamples + normalizes."""
        x = torch.nn.functional.interpolate(x_32, size=(224, 224),
                                             mode="bilinear", align_corners=False)
        return (x - imagenet_mean) / imagenet_std

    results_cls = np.zeros((args.n_alphas, N))
    results_patch = np.zeros((args.n_alphas, N))

    for ai, a in enumerate(alphas):
        mixed = (1 - a) * all_imgs + a * noise
        x = prep(mixed).to(device)
        cls_feat = dinov2_cls_backbone(model, x).cpu().numpy()
        patch_feat = dinov2_patch_mean(model, x).cpu().numpy()
        results_cls[ai] = np.linalg.norm(cls_feat, axis=1)
        results_patch[ai] = np.linalg.norm(patch_feat, axis=1)
        print(f"α={a:.2f}  μ‖CLS‖={results_cls[ai].mean():.3f}  "
              f"μ‖patch_mean‖={results_patch[ai].mean():.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.model

    # === Plot 1: overall norm trajectory ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, arr, name in [(axes[0], results_cls, "CLS token"),
                          (axes[1], results_patch, "mean patch token")]:
        mean = arr.mean(axis=1); std = arr.std(axis=1)
        ax.plot(alphas, mean, "o-", linewidth=2)
        ax.fill_between(alphas, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Noise fraction α")
        ax.set_ylabel("Feature norm")
        ax.set_title(f"{args.model}  —  {name}  (d={arr.shape[1]})")
        ax.grid(alpha=0.3)
    fig.suptitle(f"DINOv2 noise response (unnormalized features)", fontsize=13)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, f"noise_dinov2_{tag}.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out1}")

    # === Plot 2: per-class trajectories for CLS ===
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    cmap = plt.cm.tab10
    for c in range(10):
        mask = all_labels == c
        mean_c = results_cls[:, mask].mean(axis=1)
        ax.plot(alphas, mean_c, "o-", color=cmap(c), markersize=4,
                label=CIFAR_CLASSES[c], alpha=0.85)
    ax.set_xlabel("Noise fraction α")
    ax.set_ylabel("CLS token norm")
    ax.set_title(f"{args.model} CLS norm per class")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, f"noise_dinov2_perclass_{tag}.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()

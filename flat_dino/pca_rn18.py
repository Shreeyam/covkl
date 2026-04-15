"""PCA RGB visualization of ResNet-18 spatial features on CIFAR-10."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from sklearn.decomposition import PCA

from flat_dino.mini_experiment import ResNet18CIFAR


@torch.no_grad()
def spatial_features(model, img_batch):
    """Run the ResNet-18 backbone up to but not including avgpool.
    Returns (B, 512, 4, 4) feature maps for 32x32 input."""
    b = model.backbone
    x = b.conv1(img_batch)
    x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
    x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
    return x  # (B, 512, H', W')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--out-dir", default="./results/mini")
    parser.add_argument("--cpu", action="store_true")
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
    ds = torchvision.datasets.CIFAR10("./data", train=False, download=False)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(ds), args.n_images, replace=False)

    raw_imgs = []
    tensor_imgs = []
    for i in idx:
        img, _ = ds[i]
        raw_imgs.append(np.array(img))
        tensor_imgs.append(eval_transform(img))
    tensor_imgs = torch.stack(tensor_imgs).to(device)

    feat_maps = spatial_features(model, tensor_imgs)  # (B, 512, 4, 4)
    B, C, H, W = feat_maps.shape
    feats_flat = feat_maps.permute(0, 2, 3, 1).reshape(B * H * W, C).cpu().numpy()

    print(f"Fitting PCA on {feats_flat.shape[0]} patches × {C} dims...")
    pca = PCA(n_components=3)
    proj = pca.fit_transform(feats_flat)
    print(f"Explained variance: {pca.explained_variance_ratio_}")

    # Normalize each PC to [0, 1] globally so colors are comparable
    proj_norm = (proj - proj.min(axis=0)) / (proj.max(axis=0) - proj.min(axis=0) + 1e-8)
    rgb_maps = proj_norm.reshape(B, H, W, 3)

    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, B, figsize=(2 * B, 4.5))
    for i in range(B):
        axes[0, i].imshow(raw_imgs[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(rgb_maps[i], interpolation="nearest")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=11, rotation=0,
                          labelpad=40, va="center")
    axes[1, 0].set_ylabel("PCA RGB", fontsize=11, rotation=0,
                          labelpad=40, va="center")
    evr = pca.explained_variance_ratio_
    fig.suptitle(f"ResNet-18 spatial PCA (ep {ckpt['epoch']})  "
                 f"PC1={evr[0]:.2f}  PC2={evr[1]:.2f}  PC3={evr[2]:.2f}")
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"pca_rgb_rn18_ep{ckpt['epoch']}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

"""Mix CIFAR images with random noise and watch how the embedding evolves.

Hypothesis: if norm encodes confidence/canonicalness, then adding noise should
make the image less recognizable and pull the embedding toward the origin.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from flat_dino.mini_experiment import ResNet18CIFAR


CIFAR_CLASSES = ["airplane", "auto", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def embed(model, imgs_tensor, device):
    """Run student head on a batch of normalized tensor images."""
    return model(imgs_tensor.to(device)).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default="./results/mini")
    parser.add_argument("--n-per-class", type=int, default=5)
    parser.add_argument("--n-alphas", type=int, default=11)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "mps" if (torch.backends.mps.is_available() and not args.cpu) else "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ResNet18CIFAR(embed_dim=256).to(device)
    model.load_state_dict(ckpt["student"])
    model.eval()

    # Normalization constants used during training
    mean_t = torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
    std_t = torch.tensor([0.2470, 0.2435, 0.2616])[:, None, None]

    # Load the raw (unnormalized) dataset in tensor form
    to_tensor = T.ToTensor()
    raw_ds = torchvision.datasets.CIFAR10("./data", train=False, download=False)

    # Sample n_per_class images from each class
    rng = np.random.RandomState(42)
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

    # Also get a big reference set to compute a "typical noise embedding"
    # and a training-set feature bank for kNN
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=False,
                                             transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=4)
    print("Extracting train features for kNN reference...")
    train_feats, train_labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            train_feats.append(model(x.to(device)).cpu())
            train_labels.append(y)
    train_feats = torch.cat(train_feats)
    train_labels = torch.cat(train_labels).numpy()

    # Fixed noise (consistent across α values per sample)
    torch.manual_seed(0)
    noise = torch.rand_like(all_imgs)  # uniform noise in [0, 1]

    alphas = np.linspace(0.0, 1.0, args.n_alphas)

    # Collect norms and kNN predictions for each alpha
    results = {
        "norms": np.zeros((args.n_alphas, N)),
        "cos_to_clean": np.zeros((args.n_alphas, N)),
        "knn_pred": np.zeros((args.n_alphas, N), dtype=int),
        "knn_correct": np.zeros((args.n_alphas, N), dtype=bool),
    }

    clean_feats = None
    train_norm = F.normalize(train_feats, dim=-1)

    for ai, a in enumerate(alphas):
        # Mix raw [0,1] images with noise
        mixed = (1 - a) * all_imgs + a * noise
        normalized = (mixed - mean_t) / std_t
        f = embed(model, normalized, device)
        results["norms"][ai] = np.linalg.norm(f, axis=1)
        if a == 0.0:
            clean_feats = f
        # Cosine to clean
        norms_clean = np.linalg.norm(clean_feats, axis=1) + 1e-8
        norms_f = np.linalg.norm(f, axis=1) + 1e-8
        results["cos_to_clean"][ai] = (f * clean_feats).sum(axis=1) / (norms_clean * norms_f)

        # kNN classify each noised feature
        ft = F.normalize(torch.from_numpy(f), dim=-1)
        sim = ft @ train_norm.T  # (N, N_train)
        _, topk_idx = sim.topk(20, dim=-1)
        topk_labs = torch.from_numpy(train_labels)[topk_idx]
        votes = torch.zeros(N, 10)
        for c in range(10):
            votes[:, c] = (topk_labs == c).float().sum(dim=-1)
        preds = votes.argmax(dim=-1).numpy()
        results["knn_pred"][ai] = preds
        results["knn_correct"][ai] = preds == all_labels

        print(f"α={a:.2f}  μ‖z‖={results['norms'][ai].mean():.2f}  "
              f"kNN acc={results['knn_correct'][ai].mean():.3f}  "
              f"cos(clean)={results['cos_to_clean'][ai].mean():.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"ep{ckpt['epoch']}"

    # === Plot 1: norm and kNN accuracy vs alpha ===
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    mean_norm = results["norms"].mean(axis=1)
    std_norm = results["norms"].std(axis=1)
    axes[0].plot(alphas, mean_norm, "o-", linewidth=2)
    axes[0].fill_between(alphas, mean_norm - std_norm, mean_norm + std_norm, alpha=0.2)
    axes[0].set_xlabel("Noise fraction α (0=clean, 1=pure noise)")
    axes[0].set_ylabel("Feature norm")
    axes[0].set_title("Norm shrinks as noise grows")

    mean_acc = results["knn_correct"].mean(axis=1)
    axes[1].plot(alphas, mean_acc, "o-", linewidth=2, color="coral")
    axes[1].axhline(0.1, color="gray", linestyle="--", label="random chance")
    axes[1].set_xlabel("Noise fraction α")
    axes[1].set_ylabel("kNN accuracy")
    axes[1].set_title("kNN accuracy collapses to chance")
    axes[1].legend(fontsize=8)

    mean_cos = results["cos_to_clean"].mean(axis=1)
    axes[2].plot(alphas, mean_cos, "o-", linewidth=2, color="seagreen")
    axes[2].set_xlabel("Noise fraction α")
    axes[2].set_ylabel("Cosine similarity to clean embedding")
    axes[2].set_title("Direction drifts from clean")

    fig.suptitle(f"Noise injection experiment (ep {ckpt['epoch']})", fontsize=13)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, f"noise_summary_{tag}.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out1}")

    # === Plot 2: per-sample norm and cosine trajectories ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.cm.tab10
    for i in range(N):
        c = all_labels[i]
        label = CIFAR_CLASSES[c] if i == np.where(all_labels == c)[0][0] else None
        axes[0].plot(alphas, results["norms"][:, i], "o-", color=cmap(c),
                     label=label, markersize=4, alpha=0.8)
        axes[1].plot(alphas, results["cos_to_clean"][:, i], "o-", color=cmap(c),
                     label=label, markersize=4, alpha=0.8)
    axes[0].set_xlabel("Noise fraction α")
    axes[0].set_ylabel("Feature norm ‖z(α)‖")
    axes[0].set_title("Per-sample norm trajectory")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Noise fraction α")
    axes[1].set_ylabel("cos(z(α), z(0))")
    axes[1].set_title("Per-sample cosine to clean embedding")
    axes[1].axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(f"Noise response, one sample per class (ep {ckpt['epoch']})")
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, f"noise_per_class_{tag}.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # === Plot 3: sample visualization at 3 alphas ===
    alpha_show = [0.0, 0.5, 1.0]
    alpha_idx = [np.argmin(np.abs(alphas - a)) for a in alpha_show]
    n_show = 10  # one per class
    show_idx = [np.where(all_labels == c)[0][0] for c in range(10)]

    fig, axes = plt.subplots(len(alpha_show), n_show, figsize=(2 * n_show, 2 * len(alpha_show)))
    for i, ai in enumerate(alpha_idx):
        a = alphas[ai]
        for j, idx in enumerate(show_idx):
            mixed = (1 - a) * all_imgs[idx] + a * noise[idx]
            axes[i, j].imshow(mixed.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, j].axis("off")
            title_parts = []
            if i == 0:
                title_parts.append(CIFAR_CLASSES[all_labels[idx]])
            title_parts.append(f"‖z‖={results['norms'][ai][idx]:.1f}")
            if j == 0:
                title_parts.insert(0, f"α={a:.1f}")
            axes[i, j].set_title("\n".join(title_parts), fontsize=8)
    fig.suptitle(f"Sample evolution with noise (ep {ckpt['epoch']})", fontsize=13)
    fig.tight_layout()
    out3 = os.path.join(args.out_dir, f"noise_samples_{tag}.png")
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()

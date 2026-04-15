"""Mini Gaussian DINO experiment: ConvNet on CIFAR-10."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


class SmallConvNet(nn.Module):
    """Simple ConvNet encoder for 32x32 images."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def forward_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 inputs).

    Replaces the stem 7x7 stride-2 conv + maxpool with a 3x3 stride-1 conv,
    which is standard for CIFAR-scale inputs. Followed by a projection head.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone  # outputs 512-d

        self.head = nn.Sequential(
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, embed_dim),
        )

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


class MultiCropCIFAR:
    """Two random crops of each CIFAR image."""
    def __init__(self):
        self.transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


def collate_views(batch):
    v1 = torch.stack([b[0][0] for b in batch])
    v2 = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return (v1, v2), labels


def train_and_eval(config, device, train_ds, test_ds, n_epochs=50, batch_size=256,
                   name="run", log_dir="./runs/mini"):
    """Train one configuration and return metrics over time."""
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(log_dir, name))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_views, drop_last=True,
                              persistent_workers=True)

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    eval_train = torchvision.datasets.CIFAR10("./data", train=True, download=False,
                                               transform=eval_transform)
    eval_test = torchvision.datasets.CIFAR10("./data", train=False, download=False,
                                              transform=eval_transform)
    eval_train_loader = DataLoader(eval_train, batch_size=1024, shuffle=False, num_workers=4,
                                   persistent_workers=True)
    eval_test_loader = DataLoader(eval_test, batch_size=1024, shuffle=False, num_workers=4,
                                  persistent_workers=True)

    embed_dim = 256
    arch = config.get("arch", "smallconv")
    if arch == "resnet18":
        student = ResNet18CIFAR(embed_dim).to(device)
    else:
        student = SmallConvNet(embed_dim).to(device)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False

    lr = config.get("lr", 1e-3)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    ema_decay_start = config.get("ema_decay", 0.996)
    method = config.get("method", "simdino")
    use_teacher = method == "simdino"

    history = {"loss": [], "knn_acc": [], "epoch_knn": []}
    total_steps = n_epochs * len(train_loader)
    step = 0

    for epoch in range(n_epochs):
        student.train()
        epoch_loss = 0

        for (v1, v2), _ in train_loader:
            v1, v2 = v1.to(device), v2.to(device)

            s1 = student(v1)
            s2 = student(v2)
            if use_teacher:
                with torch.no_grad():
                    t1 = teacher(v1)
                    t2 = teacher(v2)

            student_all = torch.cat([s1, s2], dim=0)

            if method == "simdino":
                # SimDINO-style simplification:
                #   alignment (MSE) + correlation rate + SigReg
                # - Correlation rate (not covariance): scale-invariant decorrelation
                # - SigReg: two-sided std regularizer pins scale to 1
                cr_eps = config.get("cr_eps", 0.5)
                gamma = config.get("gamma", 1.0)
                w_sigreg = config.get("w_sigreg", 5.0)

                feats_for_reg = student_all

                # Alignment: squared Euclidean distance, cross-view (raw features)
                loss_pred = 0.5 * (F.mse_loss(s1, t2.detach()) +
                                   F.mse_loss(s2, t1.detach())) / 2

                # Correlation rate: code rate of per-dim-unit-variance features.
                # Ẑ = (Z - μ) / σ, R_corr = (1/2) logdet(I + (d/ε²) Ẑ^T Ẑ / N).
                N_batch, D = feats_for_reg.shape
                centered = feats_for_reg - feats_for_reg.mean(dim=0)
                std_per_dim = centered.std(dim=0, unbiased=False) + 1e-4
                normed = centered / std_per_dim
                corr = (normed.T @ normed) / N_batch
                I_D = torch.eye(D, device=device)
                sign, logabsdet = torch.linalg.slogdet(I_D + (D / cr_eps**2) * corr)
                R_corr = 0.5 * logabsdet

                # SigReg: two-sided MSE on std toward target 1
                loss_sigreg = torch.mean((std_per_dim - 1.0) ** 2)

                loss_var = loss_sigreg  # for logging
                loss_decorr = -R_corr   # for logging (we want to maximize R)
                loss_l2 = torch.tensor(0.0, device=device)

                loss = loss_pred + w_sigreg * loss_sigreg - gamma * R_corr

            elif method == "gaussianized":
                # Covariance-KL Gaussian matching (plain Siamese, per-view):
                #   λ_align · MSE(s1, s2)
                #   + λ_μ · mean(‖z̄ᵥ‖²/d)
                #   + λ_C · mean(½(tr(Cᵥ) - logdet(Cᵥ + ρI) - d)/d)
                lam_align = config.get("lam_align", 25.0)
                lam_mu = config.get("lam_mu", 1.0)
                lam_covkl = config.get("lam_covkl", 1.0)
                rho = config.get("rho", 1e-2)

                loss_pred = F.mse_loss(s1, s2)

                def covkl_and_mu(z):
                    N, D = z.shape
                    mean = z.mean(dim=0)
                    c = z - mean
                    C = (c.T @ c) / N
                    C_rho = (1 - rho) * C + rho * torch.eye(D, device=device)
                    sign, logabsdet = torch.linalg.slogdet(C_rho)
                    covkl = 0.5 * (torch.diagonal(C_rho).sum() - logabsdet - D) / D
                    return covkl, (mean ** 2).mean()

                c1, mu1 = covkl_and_mu(s1)
                c2, mu2 = covkl_and_mu(s2)
                L_covkl = (c1 + c2) / 2
                L_mu = (mu1 + mu2) / 2

                loss_var = L_covkl
                loss_decorr = L_mu
                loss_l2 = torch.tensor(0.0, device=device)

                loss = lam_align * loss_pred + lam_mu * L_mu + lam_covkl * L_covkl

            elif method == "vicreg":
                # VICReg (Bardes et al. 2022):
                #   λ · MSE(z1, z2) + μ · variance hinge + ν · covariance off-diag
                lam = config.get("lam_invar", 25.0)
                mu = config.get("lam_var", 25.0)
                nu = config.get("lam_cov", 1.0)

                loss_pred = F.mse_loss(s1, s2)

                def var_cov_terms(z):
                    N, D = z.shape
                    z_c = z - z.mean(dim=0)
                    std = torch.sqrt(z_c.var(dim=0, unbiased=False) + 1e-4)
                    v_hinge = torch.mean(F.relu(1.0 - std))
                    cov = (z_c.T @ z_c) / (N - 1)
                    off = cov - torch.diag(torch.diagonal(cov))
                    c = (off ** 2).sum() / D
                    return v_hinge, c

                v1_h, c1 = var_cov_terms(s1)
                v2_h, c2 = var_cov_terms(s2)
                loss_var = (v1_h + v2_h) / 2
                loss_decorr = (c1 + c2) / 2

                loss = lam * loss_pred + mu * loss_var + nu * loss_decorr
                loss_l2 = torch.tensor(0.0, device=device)

            elif method == "barlow":
                # Barlow Twins (Zbontar et al. 2021):
                #   on-diagonal (1 - C_ii)^2 + λ · off-diagonal C_ij^2
                # C is the cross-correlation of batch-normalized z1, z2.
                lam_bt = config.get("lam_bt", 5e-3)
                N, D = s1.shape

                def bn(z):
                    return (z - z.mean(dim=0)) / (z.std(dim=0, unbiased=False) + 1e-4)

                z1n, z2n = bn(s1), bn(s2)
                C = (z1n.T @ z2n) / N
                on_diag = ((torch.diagonal(C) - 1.0) ** 2).sum()
                off = C - torch.diag(torch.diagonal(C))
                off_diag = (off ** 2).sum()

                loss_pred = on_diag       # for logging: alignment on diagonal
                loss_decorr = off_diag    # for logging: redundancy reduction
                loss_var = torch.tensor(0.0, device=device)
                loss_l2 = torch.tensor(0.0, device=device)

                loss = on_diag + lam_bt * off_diag

            else:
                raise ValueError(f"unknown method: {method}")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            optimizer.step()

            if use_teacher:
                progress = step / max(total_steps, 1)
                ema_decay = 1.0 - (1.0 - ema_decay_start) * (1 + np.cos(np.pi * progress)) / 2
                with torch.no_grad():
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.data.mul_(ema_decay).add_(ps.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()
            step += 1

            # Per-step TB logging
            if step % 20 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/prediction", loss_pred.item(), step)
                writer.add_scalar("train/variance", loss_var.item(), step)
                writer.add_scalar("train/decorrelation", loss_decorr.item(), step)
                writer.add_scalar("train/l2_compression", loss_l2.item(), step)

        scheduler.step()
        epoch_loss /= len(train_loader)
        history["loss"].append(epoch_loss)
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch + 1)

        # kNN every 5 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            knn_acc = eval_knn(student, eval_train_loader, eval_test_loader, device)
            history["knn_acc"].append(knn_acc)
            history["epoch_knn"].append(epoch + 1)
            writer.add_scalar("eval/knn_accuracy", knn_acc, epoch + 1)
            print(f"  Epoch {epoch+1}/{n_epochs} | loss={epoch_loss:.4f} | kNN={knn_acc:.4f}")

            # Early stop: if at epoch 10 kNN is worse than initial, kill this run
            if (epoch + 1) == 10 and len(history["knn_acc"]) >= 2:
                initial_knn = history["knn_acc"][0]
                if knn_acc < initial_knn:
                    print(f"  [early stop] kNN at epoch 10 ({knn_acc:.4f}) < initial ({initial_knn:.4f})")
                    history["early_stopped"] = True
                    break
        else:
            print(f"  Epoch {epoch+1}/{n_epochs} | loss={epoch_loss:.4f}")

        # Periodic checkpoint save (every 20 epochs) — numbered paths so
        # the full training trajectory is preserved.
        if (epoch + 1) % 20 == 0:
            ckpt_dir = "./checkpoints/mini"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{name}_ep{epoch+1}.pth")
            torch.save({
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "config": config,
                "epoch": epoch + 1,
            }, ckpt_path)

    # Save final checkpoint
    ckpt_dir = "./checkpoints/mini"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pth")
    torch.save({
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "config": config,
        "history": {"loss": history["loss"], "knn_acc": history["knn_acc"],
                    "epoch_knn": history["epoch_knn"]},
        "epoch": epoch + 1,
    }, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    # Final features for UMAP
    features, labels = extract_features(student, eval_test_loader, device)
    history["final_features"] = features
    history["labels"] = labels
    writer.close()

    return history


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labs = [], []
    for images, labels in loader:
        f = model.forward_features(images.to(device))
        feats.append(f.cpu())
        labs.append(labels)
    return torch.cat(feats).numpy(), torch.cat(labs).numpy()


@torch.no_grad()
def eval_knn(model, train_loader, test_loader, device, k=20):
    model.eval()
    train_f, train_l = extract_features(model, train_loader, device)
    test_f, test_l = extract_features(model, test_loader, device)

    train_f = torch.from_numpy(train_f)
    test_f = torch.from_numpy(test_f)
    train_f = F.normalize(train_f, dim=-1)
    test_f = F.normalize(test_f, dim=-1)

    correct = 0
    for i in range(0, len(test_f), 1024):
        chunk = test_f[i:i+1024]
        sim = chunk @ train_f.T
        topk_sim, topk_idx = sim.topk(k, dim=-1)
        topk_labels = torch.from_numpy(train_l)[topk_idx]
        votes = torch.zeros(chunk.shape[0], 10)
        for c in range(10):
            votes[:, c] = ((topk_labels == c).float() * topk_sim).sum(dim=-1)
        correct += (votes.argmax(dim=-1) == torch.from_numpy(test_l[i:i+1024])).sum().item()

    model.train()
    return correct / len(test_l)


def sweep():
    """Hyperparameter sweep for Exp+SigReg."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mode", type=str, default="sweep", choices=["sweep", "compare"])
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Subset of config names to run (default: all)")
    args = parser.parse_args()

    if torch.backends.mps.is_available() and not args.cpu:
        device = "mps"
    elif torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    torchvision.datasets.CIFAR10("./data", train=True, download=True)
    torchvision.datasets.CIFAR10("./data", train=False, download=True)
    train_ds = torchvision.datasets.CIFAR10("./data", train=True, transform=MultiCropCIFAR())
    test_ds = torchvision.datasets.CIFAR10("./data", train=False)

    # SimDINO-style (correlation rate + SigReg). Baseline to beat: 65.3% (ema=0.99).
    simdino_base = dict(method="simdino", lr=1e-3, ema_decay=0.99)

    def sim(**overrides):
        defaults = dict(cr_eps=0.5, gamma=1.0, w_sigreg=5.0)
        return {**simdino_base, **defaults, **overrides}

    configs = {
        # Best SimDINO config (reference): γ=1e-3, ε=0.5, w_sigreg=1.0
        "SimDINO_rn18_long": sim(gamma=1e-3, w_sigreg=1.0, arch="resnet18"),
        # VICReg (Bardes et al. 2022): λ=25, μ=25, ν=1
        "VICReg_rn18": dict(method="vicreg", lr=1e-3, arch="resnet18",
                            lam_invar=25.0, lam_var=25.0, lam_cov=1.0),
        # Barlow Twins (Zbontar et al. 2021): λ=5e-3 on off-diagonal
        "Barlow_rn18": dict(method="barlow", lr=1e-3, arch="resnet18",
                            lam_bt=5e-3),
        # Covariance-KL Gaussian matching (§3.3 of paper): single cov-KL term
        # replacing correlation-rate + SigReg, plus explicit mean penalty.
        # Plain Siamese (no EMA teacher), per-view covariance, VICReg-style
        # balance (alignment weighted 25× heavier than regularizers).
        "CovKL_rn18": dict(method="gaussianized", lr=1e-3, arch="resnet18",
                           lam_align=25.0, lam_mu=1.0, lam_covkl=1.0, rho=1e-2),
        # v2: drop the mean penalty (was contributing zero) and double alignment
        # weight (invariance was underweighted relative to CovKL).
        "CovKL_v2_rn18": dict(method="gaussianized", lr=1e-3, arch="resnet18",
                              lam_align=50.0, lam_mu=0.0, lam_covkl=1.0, rho=1e-2),
        # v3: halve alignment weight (give CovKL more voice), drop mean penalty.
        "CovKL_v3_rn18": dict(method="gaussianized", lr=1e-3, arch="resnet18",
                              lam_align=12.5, lam_mu=0.0, lam_covkl=1.0, rho=1e-2),
    }

    if args.methods:
        configs = {k: v for k, v in configs.items() if k in args.methods}
        missing = set(args.methods) - set(configs.keys())
        if missing:
            raise ValueError(f"unknown configs: {missing}")

    results = {}
    for name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Training: {name} | {config}")
        print(f"{'='*50}")
        results[name] = train_and_eval(config, device, train_ds, test_ds,
                                        n_epochs=args.epochs, batch_size=args.batch_size,
                                        name=f"sweep_{name}", log_dir="./runs/sweep")

    # Summary plot
    out_dir = "./results/sweep"
    os.makedirs(out_dir, exist_ok=True)

    import scienceplots
    plt.style.use(["science", os.path.join(os.path.dirname(__file__), "..", "shreeyam.mplstyle")])

    # Group by sweep type
    groups = {"ResNet-18": list(configs.keys())}

    fig, axes = plt.subplots(1, len(groups), figsize=(5, 3.5), squeeze=False)
    axes = axes[0]

    for ax, (group_name, keys) in zip(axes, groups.items()):
        for name in keys:
            if name in results:
                h = results[name]
                ax.plot(h["epoch_knn"], h["knn_acc"], "o-", markersize=3, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("kNN Accuracy")
        ax.set_title(group_name)
        ax.legend(fontsize=6)

    fig.suptitle("Exp+SigReg Hyperparameter Sweep (CIFAR-10)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_knn.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Bar chart of final kNN
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    names = list(results.keys())
    accs = [results[n]["knn_acc"][-1] for n in names]
    bars = ax.barh(names, accs)
    ax.set_xlabel("Final kNN Accuracy")
    ax.set_title("Exp+SigReg Sweep — Final kNN")
    ax.set_xlim(min(accs) - 0.05, max(accs) + 0.02)
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_final.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # UMAP + norm dist for each config
    for name, h in results.items():
        feats = h["final_features"]
        labels = h["labels"]
        # Save features to disk
        np.savez(os.path.join(out_dir, f"features_{name}.npz"), features=feats, labels=labels)

        print(f"Fitting UMAP for {name}...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedded = reducer.fit_transform(feats)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        sc = axes[0].scatter(embedded[:, 0], embedded[:, 1], s=2, alpha=0.5,
                             c=labels, cmap="tab10")
        axes[0].set_title(f"{name} — UMAP of CIFAR-10 val (kNN={h['knn_acc'][-1]:.3f})")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        norms = np.linalg.norm(feats, axis=1)
        axes[1].hist(norms, bins=50, alpha=0.7, edgecolor="black")
        axes[1].set_title(f"Feature norms: mean={norms.mean():.2f}  std={norms.std():.2f}")
        axes[1].set_xlabel("L2 norm")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"umap_{name}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\nResults saved to {out_dir}/")
    print("\nFinal kNN accuracies:")
    for name in sorted(results.keys(), key=lambda n: results[n]["knn_acc"][-1], reverse=True):
        print(f"  {name}: {results[name]['knn_acc'][-1]:.4f}")


if __name__ == "__main__":
    sweep()

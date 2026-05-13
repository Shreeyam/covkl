"""Single-run SSL training with periodic kNN evaluation."""

from __future__ import annotations

import copy
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from covkl.data import build_loaders
from covkl.eval import eval_knn, extract_features
from covkl.losses import NEEDS_TEACHER, compute_loss
from covkl.models import build_encoder


def train_and_eval(
    config: dict,
    *,
    device: str,
    data_root: str = "./data",
    n_epochs: int = 50,
    batch_size: int = 256,
    num_workers: int = 4,
    name: str = "run",
    log_dir: str = "./runs/mini",
    ckpt_dir: str = "./checkpoints/mini",
    knn_every: int = 10,
    ckpt_every: int = 20,
    early_stop_at: Optional[int] = 10,
):
    """Train one configuration end-to-end.

    Returns a history dict with ``loss``, ``knn_acc``, ``epoch_knn``,
    ``final_features``, ``labels``, and ``early_stopped``.
    """
    writer = SummaryWriter(log_dir=os.path.join(log_dir, name))
    train_loader, eval_train_loader, eval_test_loader = build_loaders(
        data_root, batch_size=batch_size, num_workers=num_workers,
    )

    method = config.get("method", "covkl")
    embed_dim = config.get("embed_dim", 256)
    arch = config.get("arch", "smallconv")
    student = build_encoder(arch, embed_dim).to(device)

    use_teacher = method in NEEDS_TEACHER
    if use_teacher:
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad = False
    else:
        teacher = None

    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.04)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    ema_decay_start = config.get("ema_decay", 0.996)
    grad_clip = config.get("grad_clip", 3.0)

    history = {"loss": [], "knn_acc": [], "epoch_knn": [], "early_stopped": False}
    total_steps = n_epochs * len(train_loader)
    step = 0

    for epoch in range(n_epochs):
        student.train()
        epoch_loss = 0.0

        for (v1, v2), _ in train_loader:
            v1, v2 = v1.to(device), v2.to(device)
            s1 = student(v1)
            s2 = student(v2)

            extra = {}
            if use_teacher:
                with torch.no_grad():
                    extra["t1"] = teacher(v1)
                    extra["t2"] = teacher(v2)

            loss_kwargs = {k: v for k, v in config.items() if k != "method"}
            loss, parts = compute_loss(method, s1, s2, **extra, **loss_kwargs)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()

            if use_teacher:
                progress = step / max(total_steps, 1)
                ema_decay = 1.0 - (1.0 - ema_decay_start) * (1 + np.cos(np.pi * progress)) / 2
                with torch.no_grad():
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.data.mul_(ema_decay).add_(ps.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()
            step += 1

            if step % 20 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                for k, v in parts.items():
                    writer.add_scalar(f"train/{k}", v.item(), step)

        scheduler.step()
        epoch_loss /= len(train_loader)
        history["loss"].append(epoch_loss)
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch + 1)

        is_eval = (epoch + 1) % knn_every == 0 or epoch == 0
        if is_eval:
            knn_acc = eval_knn(student, eval_train_loader, eval_test_loader, device)
            history["knn_acc"].append(knn_acc)
            history["epoch_knn"].append(epoch + 1)
            writer.add_scalar("eval/knn_accuracy", knn_acc, epoch + 1)
            print(f"  Epoch {epoch+1}/{n_epochs} | loss={epoch_loss:.4f} | kNN={knn_acc:.4f}")

            if (
                early_stop_at is not None
                and (epoch + 1) == early_stop_at
                and len(history["knn_acc"]) >= 2
                and knn_acc < history["knn_acc"][0]
            ):
                print(f"  [early stop] kNN at epoch {early_stop_at} ({knn_acc:.4f}) "
                      f"< initial ({history['knn_acc'][0]:.4f})")
                history["early_stopped"] = True
                break
        else:
            print(f"  Epoch {epoch+1}/{n_epochs} | loss={epoch_loss:.4f}")

        if (epoch + 1) % ckpt_every == 0:
            _save_ckpt(ckpt_dir, f"{name}_ep{epoch+1}.pth", student, teacher, config, epoch + 1)

    _save_ckpt(
        ckpt_dir, f"{name}.pth", student, teacher, config, epoch + 1,
        history={k: history[k] for k in ("loss", "knn_acc", "epoch_knn")},
    )

    features, labels = extract_features(student, eval_test_loader, device)
    history["final_features"] = features
    history["labels"] = labels
    writer.close()
    return history


def _save_ckpt(ckpt_dir, fname, student, teacher, config, epoch, history=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, fname)
    payload = {
        "student": student.state_dict(),
        "config": config,
        "epoch": epoch,
    }
    if teacher is not None:
        payload["teacher"] = teacher.state_dict()
    if history is not None:
        payload["history"] = history
    torch.save(payload, path)
    if history is not None:
        print(f"  Saved checkpoint: {path}")


def select_device(prefer: str = "auto") -> str:
    """Pick the best available torch device. ``prefer`` may be auto/cpu/cuda/mps."""
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

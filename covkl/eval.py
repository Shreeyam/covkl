"""Feature extraction and weighted kNN evaluation on CIFAR-10."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(model, loader, device):
    """Return (features, labels) from ``model.forward_features`` over a loader."""
    model.eval()
    feats, labs = [], []
    for images, labels in loader:
        f = model.forward_features(images.to(device))
        feats.append(f.cpu())
        labs.append(labels)
    return torch.cat(feats).numpy(), torch.cat(labs).numpy()


@torch.no_grad()
def eval_knn(model, train_loader, test_loader, device, k: int = 20, num_classes: int = 10):
    """Cosine-weighted kNN top-1 accuracy on CIFAR-10."""
    train_f, train_l = extract_features(model, train_loader, device)
    test_f, test_l = extract_features(model, test_loader, device)

    train_f = F.normalize(torch.from_numpy(train_f), dim=-1)
    test_f = F.normalize(torch.from_numpy(test_f), dim=-1)
    train_l_t = torch.from_numpy(train_l)

    correct = 0
    for i in range(0, len(test_f), 1024):
        chunk = test_f[i:i + 1024]
        sim = chunk @ train_f.T
        topk_sim, topk_idx = sim.topk(k, dim=-1)
        topk_labels = train_l_t[topk_idx]
        votes = torch.zeros(chunk.shape[0], num_classes)
        for c in range(num_classes):
            votes[:, c] = ((topk_labels == c).float() * topk_sim).sum(dim=-1)
        correct += (votes.argmax(dim=-1) == torch.from_numpy(test_l[i:i + 1024])).sum().item()

    model.train()
    return correct / len(test_l)

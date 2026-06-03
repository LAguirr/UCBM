"""Notebook-style MNIST backbone utilities.

This module extracts the backbone used by ``UCBM_MnistTree.ipynb`` into reusable
PyTorch components so the notebook and the Python project can share the same
feature extractor, classifier head, and concept projection helpers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


NUM_CLASSES = 10


class BackboneG(nn.Module):
    """Spatial feature extractor used by the notebook pipeline.

    The module preserves a 64-channel 7x7 feature map so concept discovery can
    operate over spatial descriptors before global average pooling.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.feat_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x


class BackboneH(nn.Module):
    """Linear classifier head over global-average-pooled features."""

    def __init__(self, feat_dim: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat.mean(dim=[2, 3]))


class FullModel(nn.Module):
    """Convenience wrapper combining ``BackboneG`` and ``BackboneH``."""

    def __init__(self, g: nn.Module, h: nn.Module):
        super().__init__()
        self.g = g
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(self.g(x))


def build_backbone(num_classes: int = NUM_CLASSES) -> tuple[BackboneG, BackboneH, FullModel]:
    """Create the notebook-style backbone pair and the wrapped model."""

    g_net = BackboneG()
    h_net = BackboneH(g_net.feat_dim, num_classes=num_classes)
    full_model = FullModel(g_net, h_net)
    return g_net, h_net, full_model


@torch.no_grad()
def collect_gap_embeddings(loader, g: nn.Module, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect global-average-pooled embeddings and labels from a loader."""

    embeddings = []
    labels = []
    g.eval()
    for imgs, target in loader:
        feat = g(imgs.to(device))
        embeddings.append(feat.mean(dim=[2, 3]).cpu())
        labels.append(target.cpu())
    return torch.cat(embeddings), torch.cat(labels)


def project_to_concept_space(
    embeddings: torch.Tensor,
    concept_bank: torch.Tensor,
    normalize_embeddings: bool = True,
) -> torch.Tensor:
    """Project GAP embeddings onto a concept bank.

    The notebook pipeline uses L2-normalized GAP features and L2-normalized
    concept vectors, producing non-negative concept projections when the inputs
    are non-negative.
    """

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings @ concept_bank


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader, device: torch.device | str) -> float:
    """Compute classification accuracy for a model on a loader."""

    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        correct += (logits.argmax(1) == labels.to(device)).sum().item()
        total += labels.size(0)
    return correct / total


def train_notebook_backbone(
    g: BackboneG,
    h: BackboneH,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device | str,
    epochs: int = 8,
    lr: float = 1e-3,
):
    """Train the notebook-style backbone with a shared optimizer over g and h."""

    g = g.to(device)
    h = h.to(device)
    full_model = FullModel(g, h).to(device)
    optimizer = optim.Adam(full_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        full_model.train()
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = full_model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_acc = correct / total
        val_acc = evaluate_accuracy(full_model, val_loader, device)
        print(f"  Backbone epoch {epoch}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    test_acc = evaluate_accuracy(full_model, test_loader, device)
    print(f"  Backbone test_acc={test_acc:.4f}")
    return full_model

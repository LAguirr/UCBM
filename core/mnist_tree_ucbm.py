"""Three-phase UCBM training utilities ported from the notebook.

Provides the UCBMClassifier with offset-based gating and helper routines to
train the three phases (dense probe, gate-enabled transition, sparse fine-tune).
Also includes a convenience runner that collects GAP embeddings via a provided
backbone and projects them onto a concept bank for training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from typing import Optional


class UCBMClassifier(nn.Module):
    def __init__(self, n_concepts: int, n_classes: int, dropout_p: float = 0.0):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(n_concepts))
        self.linear = nn.Linear(n_concepts, n_classes, bias=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def gate(self, proj: torch.Tensor) -> torch.Tensor:
        return F.relu(proj - self.offset)

    def forward(self, proj: torch.Tensor, use_gate: bool = True):
        pi = self.gate(proj) if use_gate else proj
        logits = self.linear(self.dropout(pi))
        return logits, pi


def elastic_net(t: torch.Tensor, alpha: float = 0.99) -> torch.Tensor:
    return (1 - alpha) * 0.5 * (t**2).sum() + alpha * t.abs().sum()


@torch.no_grad()
def cbm_accuracy(clf: nn.Module, P: torch.Tensor, y: torch.Tensor, device: str = "cpu", use_gate: bool = True) -> float:
    clf.eval()
    loader = DataLoader(TensorDataset(P, y), batch_size=512)
    correct = total = 0
    for pb, lb in loader:
        logits, _ = clf(pb.to(device), use_gate=use_gate)
        correct += (logits.argmax(1) == lb.to(device)).sum().item()
        total += lb.size(0)
    return correct / total


@torch.no_grad()
def mean_active(clf: nn.Module, P: torch.Tensor, device: str = "cpu", thr: float = 1e-5) -> float:
    clf.eval()
    loader = DataLoader(TensorDataset(P, torch.zeros(len(P))), batch_size=512)
    counts = []
    for pb, _ in loader:
        _, pi = clf(pb.to(device))
        counts.append((pi.abs() > thr).float().sum(1).cpu())
    return torch.cat(counts).mean().item()


def run_phase(clf: nn.Module, P_tr: torch.Tensor, y_tr: torch.Tensor, P_test: torch.Tensor, y_test: torch.Tensor,
              lr: float, epochs: int, lam_g: float, lam_w: float, use_gate: bool, device: str, label: str):
    loader = DataLoader(TensorDataset(P_tr, y_tr), batch_size=256, shuffle=True)
    opt = optim.Adam(clf.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\n  [{label}]  lr={lr}  lam_gate={lam_g}  lam_w={lam_w}  gate={use_gate}")
    for epoch in range(1, epochs + 1):
        clf.train()
        tot = n = 0
        for pb, lb in loader:
            pb, lb = pb.to(device), lb.to(device)
            opt.zero_grad()
            logits, pi = clf(pb, use_gate=use_gate)
            loss = F.cross_entropy(logits, lb)
            if lam_g > 0:
                loss = loss + lam_g * elastic_net(pi)
            if lam_w > 0:
                loss = loss + lam_w * elastic_net(clf.linear.weight)
            loss.backward()
            opt.step()
            tot += loss.item() * pb.size(0)
            n += pb.size(0)
        sched.step()
        if epoch % 5 == 0 or epoch == epochs:
            acc = cbm_accuracy(clf, P_test, y_test, device=device, use_gate=use_gate)
            act = mean_active(clf, P_test, device=device) if use_gate else P_tr.shape[1]
            off = clf.offset.data.mean().item()
            print(f"    ep {epoch:3d}/{epochs}  loss={tot/n:.4f}  test={acc:.4f}  active={act:.1f}  offset_mean={off:.4f}")


def train_three_phase_ucbm(
    backbone_g: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    concept_bank: torch.Tensor,
    device: str = "cpu",
    NUM_CONCEPTS: Optional[int] = None,
    NUM_CLASSES: int = 10,
    TOP_K_PATCHES: int = 6,
    # phase hyperparams
    PHASE1_EPOCHS: int = 20,
    PHASE2_EPOCHS: int = 10,
    PHASE3_EPOCHS: int = 30,
    LR_PHASE1: float = 1e-3,
    LR_PHASE2: float = 5e-4,
    LR_PHASE3: float = 1e-4,
    LAM_GATE: float = 5e-5,
    LAM_W: float = 5e-6,
    DROPOUT_P: float = 0.1,
):
    """High-level runner that collects GAP embeddings, projects them and runs the three phases.

    Returns the trained UCBMClassifier and a small metrics dict.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone_g.eval()

    # collect GAP embeddings
    def collect(loader):
        embs = []
        labs = []
        with torch.no_grad():
            for imgs, labels in loader:
                feat = backbone_g(imgs.to(device))
                embs.append(feat.mean(dim=[2, 3]).cpu())
                labs.append(labels)
        return torch.cat(embs), torch.cat(labs)

    A_tr, y_train = collect(train_loader)
    A_te, y_test = collect(test_loader)

    # project
    C_n = concept_bank.to(torch.float32)
    A_tr_n = F.normalize(A_tr, dim=1)
    A_te_n = F.normalize(A_te, dim=1)
    P_train = A_tr_n @ C_n
    P_test = A_te_n @ C_n

    # probe
    probe = LogisticRegression(max_iter=1000, C=10.0)
    probe.fit(P_train.numpy(), y_train.numpy())
    probe_acc = probe.score(P_test.numpy(), y_test.numpy())

    num_concepts = P_train.shape[1] if NUM_CONCEPTS is None else NUM_CONCEPTS

    ucbm = UCBMClassifier(num_concepts, NUM_CLASSES, dropout_p=DROPOUT_P).to(device)

    # Phase 1
    run_phase(ucbm, P_train, y_train, P_test, y_test, LR_PHASE1, PHASE1_EPOCHS, 0.0, 0.0, False, device, "Phase 1: no gate, no reg")
    acc_p1 = cbm_accuracy(ucbm, P_test, y_test, device=device, use_gate=False)

    # Phase 2
    run_phase(ucbm, P_train, y_train, P_test, y_test, LR_PHASE2, PHASE2_EPOCHS, 0.0, 0.0, True, device, "Phase 2: gate enabled (identity)")
    acc_p2 = cbm_accuracy(ucbm, P_test, y_test, device=device, use_gate=True)

    # Phase 3
    run_phase(ucbm, P_train, y_train, P_test, y_test, LR_PHASE3, PHASE3_EPOCHS, LAM_GATE, LAM_W, True, device, "Phase 3: sparse fine-tune")
    final_acc = cbm_accuracy(ucbm, P_test, y_test, device=device, use_gate=True)
    avg_act = mean_active(ucbm, P_test, device=device)

    metrics = {
        "probe_acc": float(probe_acc),
        "acc_p1": float(acc_p1),
        "acc_p2": float(acc_p2),
        "final_acc": float(final_acc),
        "avg_active": float(avg_act),
    }

    return ucbm, metrics

"""
UCBM + CRAFT for MNIST — Final Correct Version
================================================

History of bugs and their fixes:
──────────────────────────────────────────────
v1 (35% acc): NMF on resized patch embeddings ≠ projection space (space mismatch)
v2 (63% acc): LAM too high, k=20 too few, gate over-regularised
v3 (72% acc): offset=-1.0 introduces a constant +1 shift into every pi value,
              breaking the linear classifier. NMF projections are always >=0
              (NMF components >=0, GAP of ReLU >=0, cosine with non-neg vectors >=0),
              so offset=-1 means pi = proj + 1.0 always — the gate never closes
              AND the linear layer sees a shifted input it can't compensate for.

Root cause of v3 gap (probe=95.7%, UCBM=72.7%, active=99.8%):
  The gate was simultaneously:
    (a) always fully open (offset=-1 < proj, so relu always passes everything)
    (b) adding a constant +1.0 shift to all projections
  The linear classifier saw proj+1 instead of proj — a distribution it was
  never designed for. 23pp gap from this single initialisation mistake.

Correct design (this version):
──────────────────────────────
1. offset init = 0.0  (correct baseline since proj >= 0 always with NMF)
   → gate starts as identity (pi = proj), no shift, matches probe exactly
   → regularisation can then push offset up to create sparsity

2. Three-phase training:
   Phase 1 — no gate at all: train linear(proj) directly, matches probe setup
   Phase 2 — gate with offset=0, no regularisation: smooth transition
   Phase 3 — gate with regularisation: sparsity emerges gradually

3. The paper (Table 1) explicitly shows "UCBM w/o concept selection & dropout"
   outperforms "UCBM" because the gate trades accuracy for sparsity.
   We implement both and let you choose the operating point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES   = 10
NUM_CONCEPTS  = 50          # k: ~5× num_classes (paper uses 3–5× ratio)
TOP_K_PATCHES = 6
BATCH_SIZE    = 256
BB_EPOCHS     = 8

# Three-phase training
PHASE1_EPOCHS = 20   # no gate, no reg  → matches probe ceiling
PHASE2_EPOCHS = 10   # gate open (offset=0), no reg  → stable transition
PHASE3_EPOCHS = 30   # gate + regularisation  → sparsity fine-tune

LR_BB      = 1e-3
LR_PHASE1  = 1e-3
LR_PHASE2  = 5e-4
LR_PHASE3  = 1e-4   # small lr so regularisation nudges, not destroys

LAM_GATE   = 5e-5   # λ_π — very gentle, just enough to move offset up
LAM_W      = 5e-6   # λ_w — tiny weight sparsity
ALPHA      = 0.99
DROPOUT_P  = 0.1
NMF_SAMPLES = 10000
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("/mnt/user-data/outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 1.  Data
# ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
unnorm = transforms.Compose([
    transforms.Normalize(mean=[0.], std=[1/0.3081]),
    transforms.Normalize(mean=[-0.1307], std=[1.]),
])

train_ds     = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_ds      = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)


# ─────────────────────────────────────────────────────────────────────
# 2.  Backbone:  g (→ spatial 64×7×7)  +  h (GAP → linear)
#
#     Critical: h uses GAP directly → linear, NO intermediate fc.
#     This ensures projection space (64-dim) == NMF space (64-dim).
# ─────────────────────────────────────────────────────────────────────
class BackboneG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1    = nn.Conv2d(1,  32, 3, padding=1)
        self.conv2    = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3    = nn.Conv2d(64, 64, 3, padding=1)
        self.pool     = nn.MaxPool2d(2, 2)
        self.feat_dim = 64

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B,32,28,28)
        x = self.pool(x)             # (B,32,14,14)
        x = F.relu(self.conv2(x))   # (B,64,14,14)
        x = F.relu(self.conv3(x))   # (B,64,14,14)
        x = self.pool(x)             # (B,64, 7, 7)
        return x


class BackboneH(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.head = nn.Linear(feat_dim, NUM_CLASSES)

    def forward(self, feat):
        return self.head(feat.mean(dim=[2, 3]))   # GAP → linear


class FullModel(nn.Module):
    def __init__(self, g, h):
        super().__init__(); self.g, self.h = g, h
    def forward(self, x):
        return self.h(self.g(x))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


print("=" * 60)
print("Step 1 — Training backbone …")
print("=" * 60)
g_net = BackboneG().to(DEVICE)
h_net = BackboneH(g_net.feat_dim).to(DEVICE)
full  = FullModel(g_net, h_net).to(DEVICE)
opt   = optim.Adam(full.parameters(), lr=LR_BB)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BB_EPOCHS)

for epoch in range(1, BB_EPOCHS + 1):
    full.train()
    correct = total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        out = full(imgs)
        F.cross_entropy(out, labels).backward()
        opt.step()
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    sched.step()
    print(f"  Epoch {epoch}/{BB_EPOCHS}  acc={correct/total:.4f}")

bb_acc = evaluate(full, test_loader)
print(f"  Black-box accuracy: {bb_acc:.4f}\n")


# ─────────────────────────────────────────────────────────────────────
# 3.  CRAFT: collect spatial descriptors
#     g(x) → (1,64,7,7) → reshape to (49,64) spatial location vectors
#     Stack → (N×49, 64): the NMF input matrix
# ─────────────────────────────────────────────────────────────────────
print("Step 2 — Collecting spatial activations …")
g_net.eval()

sampled_idx  = np.random.choice(len(train_ds), NMF_SAMPLES, replace=False)
spatial_maps = []
raw_images   = []
all_vecs     = []

for idx in sampled_idx:
    img_norm, _ = train_ds[idx]
    with torch.no_grad():
        feat = g_net(img_norm.unsqueeze(0).to(DEVICE))
    feat_np = feat.squeeze(0).cpu().numpy()           # (64,7,7)
    all_vecs.append(feat_np.reshape(64, -1).T)        # (49,64)
    spatial_maps.append(feat_np)
    raw_images.append(unnorm(img_norm).squeeze().numpy())

A_nmf = np.concatenate(all_vecs, axis=0).clip(min=0)  # (N×49, 64)
print(f"  NMF input: {A_nmf.shape}")
print(f"  Value range: [{A_nmf.min():.4f}, {A_nmf.max():.4f}]")
print(f"  (All non-negative: {(A_nmf >= 0).all()} ← confirms proj will be >= 0)\n")


# ─────────────────────────────────────────────────────────────────────
# 4.  NMF  →  concept matrix C ∈ ℝ^{64×k}
# ─────────────────────────────────────────────────────────────────────
print(f"Step 3 — NMF (k={NUM_CONCEPTS}) …")
nmf   = NMF(n_components=NUM_CONCEPTS, init="nndsvda",
            max_iter=1000, random_state=SEED)
U_nmf = nmf.fit_transform(A_nmf)                              # (N×49, k)
C_raw = nmf.components_.T                                     # (64, k)
C_norm_np = C_raw / (np.linalg.norm(C_raw, axis=0, keepdims=True) + 1e-8)
C_norm    = torch.tensor(C_norm_np, dtype=torch.float32)
print(f"  C_norm: {C_norm.shape},  col norms ~ 1: {np.linalg.norm(C_norm_np, axis=0).mean():.4f}\n")


# ─────────────────────────────────────────────────────────────────────
# 5.  Exemplar patches via NMF coefficient maps
# ─────────────────────────────────────────────────────────────────────
print("Step 4 — Finding exemplar patches …")
H_feat, W_feat = 7, 7
U_per_image = U_nmf.reshape(NMF_SAMPLES, H_feat, W_feat, NUM_CONCEPTS)

concept_exemplars = {}
for j in range(NUM_CONCEPTS):
    max_per_img = U_per_image[:, :, :, j].max(axis=(1, 2))
    top_imgs    = np.argsort(max_per_img)[::-1][:TOP_K_PATCHES]
    exemplars   = []
    for img_i in top_imgs:
        cmap    = U_per_image[img_i, :, :, j]
        cmap_up = F.interpolate(
            torch.tensor(cmap).unsqueeze(0).unsqueeze(0),
            size=(28, 28), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        pr, pc = np.unravel_index(cmap_up.argmax(), (28, 28))
        rf = 8
        r0 = max(0, pr-rf//2); c0 = max(0, pc-rf//2)
        r1 = min(28, r0+rf);   c1 = min(28, c0+rf)
        exemplars.append((raw_images[img_i], r0, c0, r1-r0, c1-c0))
    concept_exemplars[j] = exemplars


# ─────────────────────────────────────────────────────────────────────
# 6.  Project GAP embeddings onto concept space
#     proj = normalize(GAP(g(x))) @ C_norm  →  ℝ^k
#     Both GAP and C columns are non-negative → proj is always >= 0
# ─────────────────────────────────────────────────────────────────────
print("Step 5 — Computing concept projections …")

@torch.no_grad()
def get_gap_labels(loader, g):
    embs, labs = [], []
    for imgs, labels in loader:
        feat = g(imgs.to(DEVICE))
        embs.append(feat.mean(dim=[2, 3]).cpu())
        labs.append(labels)
    return torch.cat(embs), torch.cat(labs)

g_net.eval()
A_tr, y_train = get_gap_labels(train_loader, g_net)
A_te, y_test  = get_gap_labels(test_loader,  g_net)

def project(A, C_n):
    return F.normalize(A, dim=1) @ C_n   # (n,k), always >= 0

P_train = project(A_tr, C_norm)
P_test  = project(A_te, C_norm)

print(f"  P_train range: [{P_train.min():.4f}, {P_train.max():.4f}]")
print(f"  All non-negative: {(P_train >= 0).all().item()}  ← offset=0 is correct init")

probe = LogisticRegression(max_iter=1000, C=10.0, random_state=SEED)
probe.fit(P_train.numpy(), y_train.numpy())
probe_acc = probe.score(P_test.numpy(), y_test.numpy())
print(f"  Sanity probe acc: {probe_acc:.4f}  (ceiling UCBM should approach)\n")


# ─────────────────────────────────────────────────────────────────────
# 7.  UCBM classifier — correct implementation
#
#     KEY INSIGHT: since proj >= 0 always (NMF+cosine with non-neg vectors):
#       offset=0  →  gate = relu(proj - 0) = proj  (identity, no shift)
#       offset>0  →  gate zeros small projections  (sparsity)
#       offset<0  →  gate adds a constant shift  (WRONG, breaks linearity)
#
#     So offset must START at 0 and be pushed UP by regularisation.
# ─────────────────────────────────────────────────────────────────────
class UCBMClassifier(nn.Module):
    def __init__(self, n_concepts, n_classes, dropout_p):
        super().__init__()
        # offset=0: gate starts as identity (since proj>=0, relu(proj-0)=proj)
        # regularisation will push it positive to create sparsity
        self.offset  = nn.Parameter(torch.zeros(n_concepts))
        self.linear  = nn.Linear(n_concepts, n_classes, bias=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def gate(self, proj):
        """π(x) = max(0, proj − o).  With o=0 and proj≥0: identity."""
        return F.relu(proj - self.offset)

    def forward(self, proj, use_gate=True):
        pi     = self.gate(proj) if use_gate else proj
        logits = self.linear(self.dropout(pi))
        return logits, pi


def elastic_net(t, alpha=ALPHA):
    return (1-alpha)*0.5*(t**2).sum() + alpha*t.abs().sum()


@torch.no_grad()
def cbm_accuracy(clf, P, y, use_gate=True):
    clf.eval()
    loader = DataLoader(TensorDataset(P, y), batch_size=512)
    correct = total = 0
    for pb, lb in loader:
        logits, _ = clf(pb.to(DEVICE), use_gate=use_gate)
        correct += (logits.argmax(1) == lb.to(DEVICE)).sum().item()
        total   += lb.size(0)
    return correct / total


@torch.no_grad()
def mean_active(clf, P, thr=1e-5):
    clf.eval()
    loader = DataLoader(TensorDataset(P, torch.zeros(len(P))), batch_size=512)
    counts = []
    for pb, _ in loader:
        _, pi = clf(pb.to(DEVICE))
        counts.append((pi.abs() > thr).float().sum(1).cpu())
    return torch.cat(counts).mean().item()


def run_phase(clf, P_tr, y_tr, lr, epochs, lam_g, lam_w, use_gate, label):
    loader = DataLoader(TensorDataset(P_tr, y_tr), batch_size=256, shuffle=True)
    opt    = optim.Adam(clf.parameters(), lr=lr)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\n  [{label}]  lr={lr}  lam_gate={lam_g}  lam_w={lam_w}  gate={use_gate}")
    for epoch in range(1, epochs+1):
        clf.train()
        tot = n = 0
        for pb, lb in loader:
            pb, lb = pb.to(DEVICE), lb.to(DEVICE)
            opt.zero_grad()
            logits, pi = clf(pb, use_gate=use_gate)
            loss = F.cross_entropy(logits, lb)
            if lam_g > 0: loss = loss + lam_g * elastic_net(pi)
            if lam_w > 0: loss = loss + lam_w * elastic_net(clf.linear.weight)
            loss.backward(); opt.step()
            tot += loss.item() * pb.size(0); n += pb.size(0)
        sched.step()
        if epoch % 5 == 0 or epoch == epochs:
            acc  = cbm_accuracy(clf, P_test, y_test, use_gate=use_gate)
            act  = mean_active(clf, P_test) if use_gate else NUM_CONCEPTS
            off  = clf.offset.data.mean().item()
            print(f"    ep {epoch:3d}/{epochs}  loss={tot/n:.4f}  "
                  f"test={acc:.4f}  active={act:.1f}  offset_mean={off:.4f}")


print("Step 6 — Three-phase UCBM training …")
ucbm = UCBMClassifier(NUM_CONCEPTS, NUM_CLASSES, DROPOUT_P).to(DEVICE)

# Phase 1: no gate — identical setup to probe, should reach probe accuracy
run_phase(ucbm, P_train, y_train,
          lr=LR_PHASE1, epochs=PHASE1_EPOCHS,
          lam_g=0, lam_w=0, use_gate=False,
          label="Phase 1: no gate, no reg — matches probe")

acc_p1 = cbm_accuracy(ucbm, P_test, y_test, use_gate=False)
print(f"\n  After Phase 1: {acc_p1:.4f}  (probe={probe_acc:.4f})")

# Phase 2: enable gate (offset=0 → identity), still no regularisation
run_phase(ucbm, P_train, y_train,
          lr=LR_PHASE2, epochs=PHASE2_EPOCHS,
          lam_g=0, lam_w=0, use_gate=True,
          label="Phase 2: gate enabled (identity), no reg")

acc_p2 = cbm_accuracy(ucbm, P_test, y_test, use_gate=True)
print(f"\n  After Phase 2: {acc_p2:.4f}")

# Phase 3: gentle regularisation — pushes offset up, creates sparsity
run_phase(ucbm, P_train, y_train,
          lr=LR_PHASE3, epochs=PHASE3_EPOCHS,
          lam_g=LAM_GATE, lam_w=LAM_W, use_gate=True,
          label="Phase 3: sparse fine-tune")

final_acc = cbm_accuracy(ucbm, P_test, y_test, use_gate=True)
avg_act   = mean_active(ucbm, P_test)
print(f"\n  Black-box acc  : {bb_acc:.4f}")
print(f"  Sanity probe   : {probe_acc:.4f}")
print(f"  After phase 1  : {acc_p1:.4f}")
print(f"  After phase 2  : {acc_p2:.4f}")
print(f"  Final UCBM acc : {final_acc:.4f}")
print(f"  Active concepts: {avg_act:.1f}/{NUM_CONCEPTS} ({100*avg_act/NUM_CONCEPTS:.1f}%)")
print(f"  Offset mean    : {ucbm.offset.data.mean():.4f}\n")


# ─────────────────────────────────────────────────────────────────────
# 8.  Visualisation A: concept exemplar patches
# ─────────────────────────────────────────────────────────────────────
print("Step 7 — Plotting concept exemplars …")

SHOW_C = 12
fig, axes = plt.subplots(SHOW_C, TOP_K_PATCHES+1,
                          figsize=(2.1*(TOP_K_PATCHES+1), 2.1*SHOW_C))
fig.suptitle(f"CRAFT Concept Exemplars  (k={NUM_CONCEPTS})\n"
              "Row = concept  |  Columns = top activating image crops",
             fontsize=11, fontweight="bold", y=1.01)

for row, j in enumerate(range(SHOW_C)):
    axes[row,0].axis("off")
    axes[row,0].text(0.5, 0.5, f"c{j}", ha="center", va="center",
                     fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round", fc="#dde8f0", ec="#5588aa", lw=1.5))
    for col, (img_np, r0, c0, rh, cw) in enumerate(concept_exemplars[j]):
        ax = axes[row, col+1]
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
        ax.add_patch(mpatches.Rectangle(
            (c0-.5, r0-.5), cw, rh, lw=2, edgecolor="#e05c00", facecolor="none"))
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(f"Ex {col+1}", fontsize=8)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ucbm_craft_concepts.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved → ucbm_craft_concepts.png")


# ─────────────────────────────────────────────────────────────────────
# 9.  Visualisation B: per-digit decisions with concept crops
# ─────────────────────────────────────────────────────────────────────
print("Step 8 — Plotting explainable decisions …")

ucbm.eval()
W = ucbm.linear.weight.detach().cpu()

raw_test_ds = datasets.MNIST("./data", train=False, transform=transform)
sample_info = {}
for img_norm, lbl in raw_test_ds:
    if lbl not in sample_info:
        with torch.no_grad():
            feat = g_net(img_norm.unsqueeze(0).to(DEVICE))
            gap  = feat.mean(dim=[2,3]).cpu()
            proj = project(gap, C_norm)
        sample_info[lbl] = (unnorm(img_norm).squeeze().numpy(), proj)
    if len(sample_info) == 10: break

TOP_C = 4
fig, axes = plt.subplots(10, TOP_C+2, figsize=(2.5*(TOP_C+2), 2.5*10))
fig.suptitle(f"UCBM Explainable Decisions  (k={NUM_CONCEPTS}, acc={final_acc:.3f})",
             fontsize=12, fontweight="bold")

for ci, ct in enumerate(["Input","Contributions"]+[f"Top concept\n{i+1}" for i in range(TOP_C)]):
    axes[0, ci].set_title(ct, fontsize=7, fontweight="bold")

for row, digit in enumerate(sorted(sample_info.keys())):
    img_raw, proj = sample_info[digit]
    with torch.no_grad():
        logits, pi = ucbm(proj.to(DEVICE))
    pred    = logits.argmax(1).item()
    pi_np   = pi.cpu().squeeze().numpy()
    contrib = pi_np * W[pred].numpy()
    top_idx = np.argsort(np.abs(contrib))[::-1][:TOP_C]

    axes[row,0].imshow(img_raw, cmap="gray", vmin=0, vmax=1)
    axes[row,0].set_ylabel(f"GT={digit}\nPred={pred}", fontsize=7,
                            color="green" if pred==digit else "red")
    axes[row,0].set_xticks([]); axes[row,0].set_yticks([])

    ax_b = axes[row,1]
    clrs = ["steelblue" if v>=0 else "tomato" for v in contrib[top_idx][::-1]]
    ax_b.barh(range(TOP_C), contrib[top_idx][::-1], color=clrs)
    ax_b.set_yticks(range(TOP_C))
    ax_b.set_yticklabels([f"c{i}" for i in top_idx[::-1]], fontsize=6)
    ax_b.axvline(0, color="k", lw=0.5)
    ax_b.tick_params(labelsize=6)
    ax_b.set_xlabel(f"active={int((pi_np>1e-5).sum())}", fontsize=6)

    for ci, cid in enumerate(top_idx[::-1]):
        ax_p = axes[row, 2+ci]
        img_ex, r0, c0, rh, cw = concept_exemplars[cid][0]
        crop = img_ex[r0:r0+rh, c0:c0+cw]
        ax_p.imshow(crop, cmap="gray", vmin=0, vmax=1,
                    aspect="auto", interpolation="nearest")
        sign = "+" if contrib[cid]>=0 else ""
        ax_p.set_title(f"c{cid}\n({sign}{contrib[cid]:.2f})", fontsize=6,
                       color="steelblue" if contrib[cid]>=0 else "tomato")
        ax_p.axis("off")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ucbm_craft_decisions.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved → ucbm_craft_decisions.png\n")


# ─────────────────────────────────────────────────────────────────────
# 10. Final summary
# ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Final Summary")
print("=" * 60)
print(f"  Black-box acc        : {bb_acc:.4f}")
print(f"  Sanity probe acc     : {probe_acc:.4f}  ← UCBM ceiling")
print(f"  After phase 1 (dense): {acc_p1:.4f}")
print(f"  After phase 2 (gate) : {acc_p2:.4f}")
print(f"  Final UCBM acc       : {final_acc:.4f}")
print(f"  Gap to black-box     : {bb_acc - final_acc:.4f}")
print(f"  Gap to probe         : {probe_acc - final_acc:.4f}")
print(f"  Avg active concepts  : {avg_act:.1f}/{NUM_CONCEPTS} "
      f"({100*avg_act/NUM_CONCEPTS:.1f}%)")
print(f"  Learned offset mean  : {ucbm.offset.data.mean():.4f}")
print("=" * 60)
print()
print("Key lesson: offset must start at 0.0 (not -1.0).")
print("  NMF components >= 0 AND GAP of ReLU activations >= 0")
print("  => cosine projection proj >= 0 always")
print("  => relu(proj - 0) = proj  (identity at init, correct baseline)")
print("  => relu(proj - (-1)) = proj + 1  (constant shift, WRONG)")

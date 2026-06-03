import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from pathlib import Path

def visualize_image_concepts(model, dataset, image_index=None, top_k=4, patch_size=10):
    if image_index is None: image_index = np.random.randint(len(dataset))
    image, label = dataset[image_index]
    image_input = image.unsqueeze(0).to(model._device)
    model._backbone.eval()
    
    with torch.no_grad():
        feature_maps = model._backbone(image_input)
        global_feats = feature_maps.mean(dim=(2, 3)) if feature_maps.ndim == 4 else feature_maps
        global_feats = F.normalize(global_feats, p=2, dim=1)
        concept_scores = torch.matmul(global_feats, model._h.T).squeeze()
        top_scores, top_indices = torch.sort(concept_scores, descending=True)
        
        cams = torch.einsum('bchw,kc->bkhw', feature_maps, model._h[top_indices[:top_k]])
        cams_res = F.interpolate(cams, size=image.shape[1:], mode='bilinear')

    fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 3))
    axes[0].imshow(image.squeeze(), cmap='gray'); axes[0].set_title(f"Label: {label}"); axes[0].axis('off')

    for k in range(top_k):
        cam = cams_res[0, k]
        h_idx, w_idx = torch.argmax(cam) // image.shape[2], torch.argmax(cam) % image.shape[2]
        h_s, w_s = max(0, h_idx-patch_size//2), max(0, w_idx-patch_size//2)
        patch = image[:, h_s:h_s+patch_size, w_s:w_s+patch_size]
        axes[k+1].imshow(patch.squeeze(), cmap='gray'); axes[k+1].axis('off')
        axes[k+1].set_title(f"C{top_indices[k]}\nS:{top_scores[k]:.2f}")
    plt.show()


def plot_concept_exemplars(concept_exemplars: dict, save_path: str | Path, show_c: int = 12, top_k: int = 6):
    """Plot a grid of concept exemplar crops and save to `save_path`.

    Args:
        concept_exemplars: mapping concept_id -> list of (image_np, r0, c0, h, w)
        save_path: output file path
        show_c: number of concepts to show
        top_k: number of exemplar crops per concept
    """
    save_path = Path(save_path)
    fig, axes = plt.subplots(show_c, top_k + 1,
                              figsize=(2.1 * (top_k + 1), 2.1 * show_c))
    fig.suptitle(f"CRAFT Concept Exemplars  (k={len(concept_exemplars)})",
                 fontsize=11, fontweight="bold", y=1.01)

    for row, j in enumerate(range(show_c)):
        axes[row, 0].axis("off")
        axes[row, 0].text(0.5, 0.5, f"c{j}", ha="center", va="center",
                         fontsize=10, fontweight="bold",
                         bbox=dict(boxstyle="round", fc="#dde8f0", ec="#5588aa", lw=1.5))
        for col, (img_np, r0, c0, rh, cw) in enumerate(concept_exemplars[j]):
            ax = axes[row, col + 1]
            ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
            ax.add_patch(mpatches.Rectangle((c0 - .5, r0 - .5), cw, rh,
                                            lw=2, edgecolor="#e05c00", facecolor="none"))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Ex {col + 1}", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_ucbm_decisions(ucbm, backbone, concept_bank: torch.Tensor, concept_exemplars: dict, save_path: str | Path, device: str = 'cpu', top_c: int = 4, unorm_fn=None):
    """Generate per-digit decision visualizations similar to the notebook.

    Args:
        ucbm: trained UCBMClassifier (with `.linear.weight`)
        backbone: feature extractor (g)
        concept_bank: tensor shape (channels, k)
        concept_exemplars: mapping of exemplars
        save_path: output file
        device: device string
        top_c: number of top concepts to show per digit
        unorm_fn: optional callable to convert tensor image -> numpy
    """
    save_path = Path(save_path)
    backbone.eval()
    C_n = concept_bank.to(torch.float32)

    # sample one example per digit
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    raw_test_ds = datasets.MNIST("./data", train=False, transform=transform)
    sample_info = {}
    for img_norm, lbl in raw_test_ds:
        if lbl not in sample_info:
            with torch.no_grad():
                feat = backbone(img_norm.unsqueeze(0).to(device))
                gap = feat.mean(dim=[2, 3]).cpu()
                emb = F.normalize(gap, dim=1) @ C_n
            img_np = img_norm.squeeze().numpy() if unorm_fn is None else unorm_fn(img_norm)
            sample_info[lbl] = (img_np, emb)
        if len(sample_info) == 10:
            break

    W = ucbm.linear.weight.detach().cpu().numpy()

    fig, axes = plt.subplots(10, top_c + 2, figsize=(2.5 * (top_c + 2), 2.5 * 10))
    fig.suptitle(f"UCBM Explainable Decisions (k={C_n.shape[1]})", fontsize=12, fontweight="bold")
    for ci, ct in enumerate(["Input", "Contributions"] + [f"Top concept\n{i+1}" for i in range(top_c)]):
        axes[0, ci].set_title(ct, fontsize=7, fontweight="bold")

    for row, digit in enumerate(sorted(sample_info.keys())):
        img_raw, proj = sample_info[digit]
        with torch.no_grad():
            logits, pi = ucbm(proj.to(device))
        pred = logits.argmax(1).item()
        pi_np = pi.cpu().squeeze().numpy()
        contrib = pi_np * W[pred]
        top_idx = np.argsort(np.abs(contrib))[::-1][:top_c]

        axes[row, 0].imshow(img_raw, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"GT={digit}\nPred={pred}", fontsize=7,
                                color="green" if pred == digit else "red")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        ax_b = axes[row, 1]
        clrs = ["steelblue" if v >= 0 else "tomato" for v in contrib[top_idx][::-1]]
        ax_b.barh(range(top_c), contrib[top_idx][::-1], color=clrs)
        ax_b.set_yticks(range(top_c))
        ax_b.set_yticklabels([f"c{i}" for i in top_idx[::-1]], fontsize=6)
        ax_b.axvline(0, color="k", lw=0.5)
        ax_b.tick_params(labelsize=6)
        ax_b.set_xlabel(f"active={int((pi_np>1e-5).sum())}", fontsize=6)

        for ci, cid in enumerate(top_idx[::-1]):
            ax_p = axes[row, 2 + ci]
            img_ex, r0, c0, rh, cw = concept_exemplars[cid][0]
            crop = img_ex[r0:r0+rh, c0:c0+cw]
            ax_p.imshow(crop, cmap='gray', vmin=0, vmax=1, aspect='auto')
            sign = "+" if contrib[cid] >= 0 else ""
            ax_p.set_title(f"c{cid}\n({sign}{contrib[cid]:.2f})", fontsize=6,
                           color="steelblue" if contrib[cid] >= 0 else "tomato")
            ax_p.axis('off')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
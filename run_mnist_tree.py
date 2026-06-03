"""End-to-end runner that wires the notebook-style pipeline into the package.

Usage: run this script to discover concepts, train the three-phase UCBM and
save visualizations. It re-uses existing backbone training utilities when a
pretrained model is present.
"""

import os
from pathlib import Path
import torch
from core.backbone import Net, FeatureExtractorG, train_backbone
from core.dataset_utils import get_mnist_loaders
from core.mnist_tree_concepts import discover_concepts
from core.mnist_tree_ucbm import train_three_phase_ucbm
from utils.visualization import plot_concept_exemplars, plot_ucbm_decisions


BASE_DIR = Path(__file__).resolve().parent
models_dir = BASE_DIR / 'models'
models_dir.mkdir(parents=True, exist_ok=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_mnist_loaders(batch_size=64)

    # backbone: reuse the existing Net and FeatureExtractorG
    bb = Net()
    model_path = models_dir / 'mnist_cnnPytorch.pt'
    if model_path.exists():
        bb.load_state_dict(torch.load(model_path, map_location=device))
    else:
        bb = train_backbone(bb, train_loader, val_loader, test_loader, device)
        torch.save(bb.state_dict(), model_path)

    g = FeatureExtractorG(bb).to(device)

    # Concept discovery (spatial NMF)
    print("Discovering concepts (NMF)…")
    res = discover_concepts(
        dataset=train_ds,
        backbone=g,
        n_concepts=50,
        sample_size=5000,
        device=device,
        top_k_patches=6,
    )

    npy_path = BASE_DIR / 'craft_concept_bank.npy'
    torch.save(res.concept_bank, models_dir / 'concept_bank.pt')
    # also save numpy for compatibility
    import numpy as np
    np.save(npy_path, res.concept_bank.numpy())

    # Train three-phase UCBM
    print("Training three-phase UCBM…")
    ucbm, metrics = train_three_phase_ucbm(
        backbone_g=g,
        train_loader=train_loader,
        test_loader=test_loader,
        concept_bank=res.concept_bank,
        device=device,
    )
    print("Metrics:", metrics)

    # Visualizations
    out_dir = BASE_DIR / 'outputs'
    out_dir.mkdir(exist_ok=True)
    plot_concept_exemplars(res.exemplars, out_dir / 'ucbm_craft_concepts.png', show_c=12, top_k=6)
    plot_ucbm_decisions(ucbm, g, res.concept_bank, res.exemplars, out_dir / 'ucbm_craft_decisions.png', device=device)


if __name__ == '__main__':
    main()

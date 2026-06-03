import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from core.dataset_utils import get_mnist_loaders
from core.mnist_tree_backbone import build_backbone, train_notebook_backbone
from core.mnist_tree_concepts import discover_concepts
from core.mnist_tree_ucbm import train_three_phase_ucbm
from utils.visualization import plot_concept_exemplars, plot_ucbm_decisions


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Charging data...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_mnist_loaders(batch_size=64)
    print("Datasets charged!.")

    # Notebook-style backbone: 3 conv layers + GAP linear head
    print("Charging notebook-style backbone....")
    g, h, full_model = build_backbone(num_classes=10)
    backbone_path = MODELS_DIR / "mnist_tree_backbone.pt"
    if backbone_path.exists():
        payload = torch.load(backbone_path, map_location=device)
        g.load_state_dict(payload["g_state_dict"])
        h.load_state_dict(payload["h_state_dict"])
        print("Backbone charged!.")
    else:
        print("Training notebook-style backbone....")
        full_model = train_notebook_backbone(
            g=g,
            h=h,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=8,
            lr=1e-3,
        )
        torch.save(
            {"g_state_dict": full_model.g.state_dict(), "h_state_dict": full_model.h.state_dict()},
            backbone_path,
        )
        print(f"Saved backbone to {backbone_path}")

    g = g.to(device)

    # Concept discovery mirrors the notebook: sample spatial maps, fit NMF, build exemplars
    patch_size = 7
    n_concepts = 9
    sample_size = 500

    print(f"Discovering {n_concepts} concepts with notebook-style NMF....")
    concept_result = discover_concepts(
        dataset=train_ds,
        backbone=g,
        n_concepts=n_concepts,
        sample_size=sample_size,
        device=device,
        top_k_patches=6,
        seed=0,
    )

    concept_bank_path = BASE_DIR / "craft_concept_bank.npy"
    np.save(concept_bank_path, concept_result.concept_bank.numpy())
    print(f"Concept bank saved to {concept_bank_path}")

    # Train the notebook-style UCBM with the three phases
    print("-----------------------------------------       Training UCBM....")
    ucbm, metrics = train_three_phase_ucbm(
        backbone_g=g,
        train_loader=train_loader,
        test_loader=test_loader,
        concept_bank=concept_result.concept_bank,
        device=device,
        NUM_CONCEPTS=n_concepts,
        NUM_CLASSES=10,
        PHASE1_EPOCHS=20,
        PHASE2_EPOCHS=10,
        PHASE3_EPOCHS=30,
        LR_PHASE1=1e-3,
        LR_PHASE2=5e-4,
        LR_PHASE3=1e-4,
        LAM_GATE=5e-5,
        LAM_W=5e-6,
        DROPOUT_P=0.1,
    )
    print(json.dumps(metrics, indent=2))

    save_name = f"mnist_tree_{datetime.now().strftime('%Y_%m_%d_-_%H_%M_%S')}"
    model_dir = BASE_DIR / "Model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_g": g.state_dict(),
            "ucbm_state_dict": ucbm.state_dict(),
            "concept_bank": concept_result.concept_bank.cpu(),
            "metrics": metrics,
        },
        model_dir / "classifier.pth",
    )

    info_dict = {
        "save_name": save_name,
        "amount of concepts": n_concepts,
        "amount of classes": 10,
        "amount of samples": sample_size,
        "patch_size": patch_size,
        "train acc": metrics["final_acc"],
        "probe acc": metrics["probe_acc"],
        "phase 1 acc": metrics["acc_p1"],
        "phase 2 acc": metrics["acc_p2"],
        "avg active concepts": metrics["avg_active"],
        "learning rate": 1e-4,
        "lambda gate": 5e-5,
        "lambda w": 5e-6,
        "dropout p": 0.1,
        "normalize": True,
        "k": -1,
    }

    with open(model_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2)
    print(f"Saved information to {model_dir}")
    print("-----------------------------------------        UCBM Trained!!")

    # Notebook-style visualizations
    plot_concept_exemplars(concept_result.exemplars, OUTPUTS_DIR / "ucbm_craft_concepts.png", show_c=12, top_k=6)
    plot_ucbm_decisions(ucbm, g, concept_result.concept_bank, concept_result.exemplars, OUTPUTS_DIR / "ucbm_craft_decisions.png", device=device)
    print(f"Saved visualizations to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()

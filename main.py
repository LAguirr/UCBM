import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from core.backbone import Net, FeatureExtractorG, train_backbone
from core.ucbm_layers import UCBM
from core.dataset_utils import load_data, get_mnist_loaders
from utils.visualization import visualize_image_concepts
from craft.craft_torch import Craft
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
models_dir = script_dir / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # 1. Config & Data
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("Charging data... ")

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_mnist_loaders(batch_size=64)

    print("Datasets charged!. ")

    # 2. Backbone
    print("Charging model....")
    backbone = Net()
    models_dir = script_dir / 'models'

    model_path = models_dir /'mnist_cnnPytorch.pt'
    if os.path.exists(model_path):
        backbone.load_state_dict(torch.load(model_path, map_location=device))
        print("Model charged!. ")
    else:
        #train the model
        print("Training the model....")
        backbone = train_backbone(backbone, train_loader, val_loader,test_loader, device, epochs=1)
        print("Model trained!!")
    
    g = FeatureExtractorG(backbone).to(device)

    # 3. Concept Discovery (CRAFT)
    print("Discovering concepts with CRAFT....")
    images_batch = torch.stack([train_ds[i][0] for i in range(500)]).to(device)
    craft = Craft(input_to_latent=g, latent_to_logit=lambda x: x, number_of_concepts=15, patch_size=7, device=device)
    crops, crops_u, w = craft.fit(images_batch)
    np.save(script_dir /"craft_concept_bank.npy", w)

    print("Concepts discovered!", crops.shape, crops_u.shape, w.shape)

    # 4. Train UCBM
    print("-----------------------------------------       Training UCBM....")
    ph_cbm = UCBM(backbone=g, h=w, batch_size=64, epochs=1, lam_gate=0.01, lam_w=0.01, dropout_p=0, learning_rate=0.01, relu="ReLU", scale_mode="no", bias_mode="no", normalize=False, k=-1, device=device)
    ph_cbm.fit(train_ds, "./mnist_activations")
    print("-----------------------------------------        UCBM Trained!!")
    # 5. Visualize
    #visualize_image_concepts(ph_cbm, test_ds)
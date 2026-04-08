import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from core.backbone import Net,ClassifierH, FeatureExtractorG, train_backbone
from core.ucbm_layers import UCBM
from core.dataset_utils import load_data, get_mnist_loaders
from utils.visualization import visualize_image_concepts
from mycraft.craft_torch import Craft
import os
from pathlib import Path
import torch.nn.functional as F
from datetime import datetime
import json
from os import path, makedirs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings


BASE_DIR = Path(__file__).resolve().parent
models_dir = BASE_DIR / 'models'
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
    models_dir = BASE_DIR / 'models'

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
    h = ClassifierH(g).to(device)

    # 3. Concept Discovery (CRAFT)
    print("Discovering concepts with CRAFT....")
    images_batch = torch.stack([train_ds[i][0] for i in range(500)]).to(device)

    patch_size = 7
    n_concepts = 80
    craft = Craft(input_to_latent=g, latent_to_logit=h, number_of_concepts=n_concepts, patch_size=patch_size, device=device)
    crops, crops_u, w = craft.fit(images_batch, gradcam=True)
    np.save(BASE_DIR /"craft_concept_bank.npy", w)

    print("Concepts discovered!", crops.shape, crops_u.shape, w.shape)

    # 4. Train UCBM
    epochs =  20
    lam_gate =  0
    lam_w = 0
    dropout_p = 0.0 #0.2
    lr = 0.01
    cls_save_name= "topk_seed_0"
    scale_choose= 'learn' #'no'
    bias_choose='learn' #-- normalize_concepts
    normalize_concepts = True # Boolean
    relu='ReLU'
    k = -1
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "MNIST"
    cls_save_name = "topk_seed_0"
    h = np.load(BASE_DIR / "craft_concept_bank.npy")
    h_tensor = torch.tensor(h, dtype=torch.float32)
    h_tensor = F.normalize(h_tensor, p=2, dim=1) # Normalización L2 estricta
    h = h_tensor.numpy()

    print("-----------------------------------------       Training UCBM....")
    ph_cbm = UCBM(backbone=g, h=h, batch_size=64, epochs=epochs, lam_gate=lam_gate, lam_w=lam_w, dropout_p=dropout_p, learning_rate=lr, relu=relu, scale_mode=scale_choose, bias_mode=bias_choose, normalize=normalize_concepts, k=k, device=device)
    ph_cbm.fit(train_ds, BASE_DIR / "mnist_activations")

    save_name = cls_save_name # author says is "topk_seed_0"
    if save_name == "":
        save_name = f"class_{datetime.now().strftime('%Y_%m_%d_-_%H_%M_%S')}"
    else:
        save_name += f"-{datetime.now().strftime('%Y_%m_%d_-_%H_%M_%S')}"
    class_path = BASE_DIR / "Model" #class_path = path.join(plotter.get_classifier_path(), args.concept_data, save_name)
    makedirs(class_path, exist_ok=True)
    ph_cbm.save_to_file(class_path, "classifier.pth")

    metrics = ["acc", "auprc", "auprc_pc", "auroc"]
    act_path = BASE_DIR / "mnist_activations"
    os.makedirs(act_path, exist_ok=True)

    info_dict = ph_cbm.get_info_dict(training_data=train_ds, test_data=test_ds, act_bank_path=act_path, images_preprocessed=images_batch.shape[0], patch_size=patch_size, total_patches=crops_u.shape[0], metrics=metrics)
    print(json.dumps(info_dict, indent=2))
    with open(path.join(class_path, "info.json"), "w") as f:
        json.dump(info_dict, f, indent=2)
    print(f"Saved information to {class_path}")
    print("-----------------------------------------        UCBM Trained!!")
    
    # 5. Visualize
    visualize_image_concepts(ph_cbm, test_ds)   
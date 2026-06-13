import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from core.backbone import Net,ClassifierH, FeatureExtractorG, train_backbone
from core.dataset_utils import load_data, get_mnist_loaders
from utils.visualization import visualize_top_patches
from mycraft.craft_torch import Craft, torch_to_numpy
import os
import pickle

from pathlib import Path
import torch.nn.functional as F
from datetime import datetime
import json
from os import path, makedirs
from core.cbdt_layers import ConceptBasedDecisionTree, create_confusion_matrix_visualization, visualize_decision_journey
from core.dataset_utils import images_preprocessing
from dotenv import load_dotenv
load_dotenv()
from utils.visualization import visualize_digit_seven, explain_decision_for_digit, visualize_image_concepts_with_craft, visualize_concepts_with_crops, visualize_image_concepts
from utils.wandb_logger import log_experiment_to_wandb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings
wandb_api_key = os.getenv("WANDB_API_KEY")


BASE_DIR = Path(__file__).resolve().parent
models_dir = BASE_DIR / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

act_path = BASE_DIR / "mnist_activations"
os.makedirs(act_path, exist_ok=True)

if __name__ == "__main__":
    # 1. Config & Data
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("Charging data... ")

    full_dataset, train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_mnist_loaders(batch_size=64)

    print("Datasets charged!. ")

    # 2. Backbone
    print("Charging model....")
    backbone = Net()
    models_dir = BASE_DIR / 'models'

    model_path = models_dir /'mnist_cnnPytorch.pt'

    if os.path.exists(model_path):
        backbone.load_state_dict(torch.load(model_path, map_location=device))
        print("BACKBONE CNN Model charged!. ")
    else:
        #train the model
        print("Training backbone CNN the model....")
        backbone = train_backbone(backbone, train_loader, val_loader,test_loader, device)
        print("Model trained!!")
    

    g = FeatureExtractorG(backbone).to(device)
    h = ClassifierH(g).to(device)

    # 3. Concept Discovery (CRAFT)

    patch_size = 9
    n_concepts = 30

    print(f"Discovering {n_concepts} concepts with CRAFT....")

    images_preprocessed = images_preprocessing(full_dataset, device)

    craft = Craft(input_to_latent=g, latent_to_logit=h, number_of_concepts=n_concepts, patch_size=patch_size, device=device)
    crops, crops_u, w = craft.fit(images_preprocessed)
    crops = np.moveaxis(torch_to_numpy(crops), 1, -1)
    np.save(BASE_DIR /"craft_concept_bank.npy", w)

    print("Concepts discovered! ", crops.shape, crops_u.shape, w.shape)

    # 4. Train Tree-based Classifier (UCBM)

    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "MNIST"
    save_name = "topk_seed_0"

    try:
        h = np.load(BASE_DIR / "craft_concept_bank.npy") #this is the h.py <----------------    10 CONCEPT BANK
        print("Concepts loaded")
    except NameError:
        print("No Concepts file found", NameError)


    h_tensor = torch.tensor(h, dtype=torch.float32)
    h_tensor = F.normalize(h_tensor, p=2, dim=1) # Normalización L2 estricta
    h = h_tensor.numpy()
    act_path = BASE_DIR / "mnist_activations"


    print("="*70)
    print("Computing Concept Activations")
    print("="*70)
    concept_activations = crops_u @ w  # [n_patches, n_concepts]
    #visualize_top_patches(n_concepts, concept_activations, crops)

    print(f"✓ Crops shape: {crops.shape}")
    print(f"✓ Crops_u shape: {crops_u.shape}")
    print(f"✓ Concept activations shape: {concept_activations.shape}")
    print(f"✓ Number of concepts: {concept_activations.shape[1]}")


    print(f"✓ Training set size: {train_ds}")
    tree = ConceptBasedDecisionTree(
            backbone=g,
            h=h,
            crops=crops,  # Use training crops
            concept_activations=concept_activations,  # For extracting top patches
            act_path=act_path,
            max_depth=4,
            min_samples_split=2,
            device=device,
            batch_size=64
        )
    
    
    print(f"✓ Tree created")
    print(f"✓ Number of concepts: {tree.n_concepts}")
    print(f"✓ Patch dimensions: {tree.patch_h}x{tree.patch_w}")

    train_metrics = tree.train(train_ds)
    print(f"✓ Training complete!")
    print("\n" + "-"*70)
    print("Evaluating Tree")

    # Validation
    #val_metrics = tree.evaluate(val_ds)
    #print(f"✓ Val Accuracy: {val_metrics['accuracy']*100:.2f}%")

    # Test
    #test_metrics = tree.evaluate(test_ds)
    #print(f"✓ Test Accuracy: {test_metrics['accuracy']*100:.2f}%")

    #print("-"*70)
    #print("Preparing Visualization of Tree Structure")
    #tree.visualize_tree_structure(crops.shape[1:3])

    
    
    # Visualize
    labels_test = torch.cat([labels for _, labels in test_loader]).tolist()

    #create_confusion_matrix_visualization(test_metrics['predictions'], labels_test)

    #tree.tree.score(test_ds,labels_test),
    #data_dict =  {
    #        'train_accuracy': train_metrics['accuracy'],
    #        'val_accuracy': val_metrics['accuracy'],
    #        'test_accuracy': test_metrics['accuracy'],
    #}
      
    #visualize_decision_journey(test_ds, act_path, tree, image_index=0)

    #visualize_digit_seven(tree, test_ds ,n_examples=3, act_path = act_path)

    #explain_decision_for_digit(tree, act_path, test_ds)
    
    #visualize_image_concepts_with_craft(tree, test_ds, crops, concept_activations, image_index=None, top_k=3, patch_size=9)

    #visualize_concepts_with_crops(crops, concept_activations, top_k=5)
    
    #visualize_image_concepts(tree, test_ds, image_index=None, top_k=2, patch_size=9)

    
    #Save the model
    with open('concept_decision_tree.pkl', 'wb') as f:
        pickle.dump(tree, f)
    print("\n✓ Tree saved to concept_decision_tree.pkl")

    #Save the information about the tree and the training process
    class_path = BASE_DIR / "Model" 
    makedirs(class_path, exist_ok=True)
    metrics = ['acc']    
    info_dict = tree.get_info_dict(training_data = train_ds, test_data=test_ds, val_data = val_ds, act_bank_path=act_path, images_preprocessed=images_preprocessed.shape[0], patch_size=patch_size, total_patches=crops_u.shape[0], metrics=metrics)
    print(json.dumps(info_dict, indent=2))
    with open(path.join(class_path, "info.json"), "w") as f:
        json.dump(info_dict, f, indent=2)
    print(f"Saved information to {class_path}")

    # Log to Weights & Biases
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = f"CBM_Exp_{current_time}"
    log_experiment_to_wandb("CBDT_project", run_id, info_dict, api_key=wandb_api_key)

    print("----------------------------------------- Tree Trained!")
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import  Subset


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

def visualize_top_patches(n_concepts, concept_activations, crops): 
    concept_activations = (concept_activations - concept_activations.min(axis=0)) / (concept_activations.max(axis=0) - concept_activations.min(axis=0) + 1e-8)

    num_concepts_to_show = min(5, n_concepts)

    # 1. Create ONE giant master figure that is tall enough for all 10 concepts
    # Width is 8, Height is 2.5 inches per concept (so 8x25 for 10 concepts)
    fig = plt.figure(figsize=(8, 2.5 * num_concepts_to_show))

    # 2. Divide this master figure into vertically stacked "sub-figures"
    subfigs = fig.subfigures(num_concepts_to_show, 1)

    # Show top patches for each concept
    for concept_id in range(num_concepts_to_show):
        top_patch_ids = np.argsort(concept_activations[:, concept_id])[-5:]

        # 3. Create your perfect 1x5 grid inside the current sub-figure
        axes = subfigs[concept_id].subplots(1, 5)
                
        for idx, patch_id in enumerate(top_patch_ids[::-1]):
            axes[idx].imshow(crops[patch_id, :, :, 0], cmap='gray')
            axes[idx].set_title(f"Activation: {concept_activations[patch_id, concept_id]:.2f}", fontsize=9)
            axes[idx].axis('off')

        # 4. Add the title exactly as you had it, but attach it to the sub-figure
        subfigs[concept_id].suptitle(f"Concept {concept_id} - Top Activating Patches", fontsize=12)

    # 5. Show the single stacked master figure
    plt.show()


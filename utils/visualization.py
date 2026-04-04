import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_image_concepts(model, dataset, image_index=None, top_k=4, patch_size=1):
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
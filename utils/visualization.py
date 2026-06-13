import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import  Subset
from core.cbdt_layers import visualize_decision_journey


def visualize_image_concepts(model, dataset, image_index=None, top_k=5, patch_size=12):
    """
    Visualizes a single image and the top K concepts that activate for it,
    showing the specific patch (crop) where each concept is looking.
    """
    if image_index is None:
        image_index = np.random.randint(len(dataset))

    image, label = dataset[image_index]
    image_input = image.unsqueeze(0).to(model._device) # Add batch dim

    # Get Model Internals
    model._backbone.eval()
    concepts = model._h.to(model._device)

    with torch.no_grad():
        # Get Spatial Features (Batch, Channels, H, W)
        feature_maps = model._backbone(image_input)

        # Get Global Concept Scores
        # Pool (B, C, H, W) -> (B, C)
        if len(feature_maps.shape) == 4:
            global_feats = torch.mean(feature_maps, dim=(2, 3))
        else:
            global_feats = feature_maps

        global_feats = global_feats.flatten(1)
        global_feats = global_feats / global_feats.norm(dim=1, keepdim=True)

        # Concept Scores: (1, Num_Concepts)
        concept_scores = torch.matmul(global_feats, concepts.T).squeeze()

        # Get Spatial Activation Maps for Top Concepts
        # Sort concepts by score
        top_scores, top_indices = torch.sort(concept_scores, descending=True)
        top_scores = top_scores[:top_k]
        top_indices = top_indices[:top_k]

        # Calculate CAMs only for these top concepts
        # Einsum: (1, Channels, H, W) x (K, Channels) -> (1, K, H, W)
        relevant_concepts = concepts[top_indices]
        cams = torch.einsum('bchw,kc->bkhw', feature_maps, relevant_concepts)

        # Upsample to image size
        cams_resized = F.interpolate(cams, size=image.shape[1:], mode='bilinear', align_corners=False)

    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Plot Original Image
    ax_orig = axes[0]
    img_disp = image.permute(1, 2, 0).numpy()
    if img_disp.shape[2] == 1:
        ax_orig.imshow(img_disp.squeeze(), cmap='gray')
    else:
        ax_orig.imshow(img_disp)
    ax_orig.set_title(f"Input: Class {label}\nIdx: {image_index}", fontsize=12)
    ax_orig.axis('off')

    # Plot Top Concepts (Crops)
    for k in range(top_k):
        c_idx = top_indices[k].item()
        score = top_scores[k].item()

        # Find Hotspot (Max activation)
        cam = cams_resized[0, k, :, :]
        idx_flat = torch.argmax(cam)
        h_idx, w_idx = idx_flat // image.shape[2], idx_flat % image.shape[2]

        # Crop Patch
        h_start = max(0, h_idx.item() - patch_size // 2)
        w_start = max(0, w_idx.item() - patch_size // 2)
        h_end = min(image.shape[1], h_start + patch_size)
        w_end = min(image.shape[2], w_start + patch_size)

        patch = image[:, h_start:h_end, w_start:w_end]

        # Display Patch
        ax = axes[k + 1]
        patch_disp = patch.permute(1, 2, 0).numpy()

        if patch_disp.shape[2] == 1:
            ax.imshow(patch_disp.squeeze(), cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(patch_disp)

        ax.set_title(f"Concept {c_idx}\nScore: {score:.2f}", fontsize=10)
        ax.axis('off')

        # Add a red box on the original image to show where it looked? (Optional)
        rect = plt.Rectangle((w_start, h_start), w_end-w_start, h_end-h_start,
                             linewidth=1, edgecolor='red', facecolor='none')
        ax_orig.add_patch(rect)

    plt.tight_layout()
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

def visualize_digit_seven(tree, test_subset, n_examples=3, act_path=None):
    """
    Show decision journeys for digit 7.

    """
    test_activations_only = tree._get_concept_embeddings(test_subset, act_path, data_label="test_set_7").cpu().numpy()
    actual_test_labels = np.array([label for _, label in test_subset])
    data_dict = {
      'labels_test': actual_test_labels,
      'concept_activations_test': test_activations_only
        }

    test_labels = data_dict['labels_test']
    concept_act_test = data_dict['concept_activations_test']

    # Find indices where label is 7
    seven_indices = np.where(test_labels == 7)[0]

    print(f"Found {len(seven_indices)} examples of digit 7 in test set")

    if len(seven_indices) == 0:
        print("No digit 7 found in test set!")
        return

    # Show up to n_examples
    for example_num in range(min(n_examples, len(seven_indices))):
        idx = seven_indices[example_num]

        # Get image from test_subset
        test_image, true_label = test_subset[idx]

        # Get concept activation
        concept_activation = concept_act_test[idx]

        # Get prediction
        pred_label = tree.predict(concept_activation.reshape(1, -1))[0]

        print(f"\n  Example {example_num + 1}:")
        print(f"    Index: {idx}")
        print(f"    True Label: {true_label}")
        print(f"    Predicted: {pred_label}")
        print(f"    Correct: {'✓' if true_label == pred_label else '✗'}")

        # Visualize the journey
        visualize_decision_journey(test_subset, act_path, tree, image_index=idx)


def explain_decision_for_digit(
    tree,
    act_path,
    test_subset,
    digit=7,
    example_num=0
):
    """
    Print detailed decision path for a specific digit example.

    """
    test_activations_only = tree._get_concept_embeddings(test_subset, act_path, data_label="test_set_7").cpu().numpy()
    actual_test_labels = np.array([label for _, label in test_subset])
    data_dict = {
        'labels_test': actual_test_labels,
        'concept_activations_test': test_activations_only
    }

    test_labels = data_dict['labels_test']
    concept_act_test = data_dict['concept_activations_test']

    # Find examples of this digit
    digit_indices = np.where(test_labels == digit)[0]

    if len(digit_indices) <= example_num:
        print(f"Only {len(digit_indices)} examples of digit {digit} available!")
        return

    idx = digit_indices[example_num]
    test_image, true_label = test_subset[idx]
    concept_activation = concept_act_test[idx]

    # Get decision path
    node_path, feature_values, prediction = tree.get_decision_path(concept_activation)

    print(f"\n{'='*70}")
    print(f"DECISION PATH FOR DIGIT {digit} (Example {example_num})")
    print(f"{'='*70}")
    print(f"\nInput: Image Index {idx}, True Label: {true_label}")
    print(f"Tree Prediction: {prediction}")
    print(f"Correct: {'✓' if true_label == prediction else '✗'}")

    print(f"\n{'-'*70}")
    print("STEP-BY-STEP DECISION PATH:")
    print(f"{'-'*70}\n")

    for step, node_idx in enumerate(node_path):
        feature_idx = tree.get_feature_at_node(node_idx)

        if feature_idx >= 0:  # Decision node
            threshold = tree.get_threshold_at_node(node_idx)
            activation = concept_activation[feature_idx]
            direction = "LEFT" if activation < threshold else "RIGHT"

            print(f"Step {step}: Node {node_idx}")
            print(f"  Concept: {feature_idx}")
            print(f"  Activation: {activation:.4f}")
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Decision: {direction} ({activation:.4f} {'<' if direction == 'LEFT' else '≥'} {threshold:.4f})")
            print()
        else:
            # Leaf node
            class_at_leaf = tree.get_node_class(node_idx)
            print(f"Step {step}: Node {node_idx} (LEAF)")
            print(f"  Predicted Class: {class_at_leaf}")
            print()


def visualize_image_concepts_with_craft(model, dataset, crops, concept_activations,
                                        image_index=None, top_k=5, patch_size=12):
    """
    Visualizes a single image from dataset and the top K concepts that activate for it.
    Shows:
    - Left: Original input image with CAM heatmap / bounding box
    - Right: Best CRAFT crops that represent each discovered concept

    Args:
        model: UCBM trained model (or ConceptBasedDecisionTree)
        dataset: Test dataset
        crops: CRAFT crops array [n_patches, H, W, C]
        concept_activations: Activation scores [n_patches, n_concepts]
        image_index: Index of image to visualize (random if None)
        top_k: Number of top concepts to display
        patch_size: Size of patches to show from input image
    """

    if image_index is None:
        image_index = np.random.randint(len(dataset))

    image, label = dataset[image_index]
    image_input = image.unsqueeze(0).to(model._device)  # Add batch dim

    # Get Model Internals
    model._backbone.eval()
    concepts = model._h.to(model._device)

    with torch.no_grad():
        # Get Spatial Features (Batch, Channels, H, W)
        feature_maps = model._backbone(image_input)

        # Get Global Concept Scores
        # Pool (B, C, H, W) -> (B, C)
        if len(feature_maps.shape) == 4:
            global_feats = torch.mean(feature_maps, dim=(2, 3))
        else:
            global_feats = feature_maps

        global_feats = global_feats.flatten(1)
        global_feats = global_feats / global_feats.norm(dim=1, keepdim=True)

        # Concept Scores: (1, Num_Concepts)
        concept_scores = torch.matmul(global_feats, concepts.T).squeeze()

        # Get Spatial Activation Maps for Top Concepts
        # Sort concepts by score
        top_scores, top_indices = torch.sort(concept_scores, descending=True)
        top_scores = top_scores[:top_k]
        top_indices = top_indices[:top_k]

        # Calculate CAMs only for these top concepts
        # Einsum: (1, Channels, H, W) x (K, Channels) -> (1, K, H, W)
        relevant_concepts = concepts[top_indices]
        cams = torch.einsum('bchw,kc->bkhw', feature_maps, relevant_concepts)

        # Upsample to image size
        cams_resized = F.interpolate(cams, size=image.shape[1:], mode='bilinear', align_corners=False)

    # Create figure: Original image + Top K concepts (2 columns: CAM + Best CRAFT crop)
    fig, axes = plt.subplots(top_k + 1, 2, figsize=(6, 3 * (top_k + 1)))

    if axes.ndim == 1:
        axes = axes.reshape(-1, 2)

    # ============================================
    # ROW 0: Original Input Image (spanning 2 columns)
    # ============================================
    ax_orig = axes[0, 0]
    img_disp = image.permute(1, 2, 0).numpy()
    if img_disp.shape[2] == 1:
        ax_orig.imshow(img_disp.squeeze(), cmap='gray')
    else:
        ax_orig.imshow(img_disp)
    ax_orig.set_title(f"Input Image: Class {label}\nIdx: {image_index}", fontsize=12, fontweight='bold')
    ax_orig.axis('off')

    # Hide the second column in first row
    axes[0, 1].axis('off')

    # ============================================
    # ROWS 1 to top_k: Concepts with CAM + Best CRAFT Crops
    # ============================================
    for k in range(top_k):
        c_idx = top_indices[k].item()
        score = top_scores[k].item()

        # -------- LEFT: CAM from input image --------
        ax_cam = axes[k + 1, 0]

        # Find Hotspot (Max activation in CAM)
        cam = cams_resized[0, k, :, :]
        idx_flat = torch.argmax(cam)
        h_idx, w_idx = idx_flat // image.shape[2], idx_flat % image.shape[2]

        # Crop Patch from input image
        h_start = max(0, h_idx.item() - patch_size // 2)
        w_start = max(0, w_idx.item() - patch_size // 2)
        h_end = min(image.shape[1], h_start + patch_size)
        w_end = min(image.shape[2], w_start + patch_size)

        patch_input = image[:, h_start:h_end, w_start:w_end]
        patch_disp = patch_input.permute(1, 2, 0).numpy()

        if patch_disp.shape[2] == 1:
            ax_cam.imshow(patch_disp.squeeze(), cmap='gray', vmin=0, vmax=1)
        else:
            ax_cam.imshow(patch_disp)

        ax_cam.set_title(f"Concept {c_idx} in Input\nModel Score: {score:.3f}", fontsize=10)
        ax_cam.axis('off')

        # Draw red rectangle showing where CAM looked
        rect = plt.Rectangle((w_start, h_start), w_end - w_start, h_end - h_start,
                             linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)

        # -------- RIGHT: Best CRAFT crop for this concept --------
        ax_craft = axes[k + 1, 1]

        # Find the crop with highest activation for this concept
        concept_crop_activations = concept_activations[:, c_idx]
        best_crop_idx = np.argmax(concept_crop_activations)
        best_crop_activation = concept_crop_activations[best_crop_idx]

        # Get the crop
        best_crop = crops[best_crop_idx]

        # Display crop
        if best_crop.ndim == 3 and best_crop.shape[2] == 1:
            ax_craft.imshow(best_crop.squeeze(), cmap='gray', vmin=0, vmax=1)
        elif best_crop.ndim == 2:
            ax_craft.imshow(best_crop, cmap='gray', vmin=0, vmax=1)
        else:
            ax_craft.imshow(best_crop)

        ax_craft.set_title(f"Best CRAFT Crop for Concept {c_idx}\nCRAFT Activation: {best_crop_activation:.3f}\nCrop #{best_crop_idx}",
                          fontsize=10, fontweight='bold')
        
        # Turn off ticks but keep spines visible for the green border
        ax_craft.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_craft.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2)
            spine.set_visible(True)

    plt.tight_layout()
    plt.show()


def visualize_concepts_with_crops(crops, concept_activations, top_k=5):
    """
    Visualizes the top K concepts by showing the actual crops with the highest
    activation scores for each concept.

    Args:
        crops: Array of image patches [n_patches, H, W, C]
        concept_activations: Array of activation scores [n_patches, n_concepts]
        top_k: Number of top concepts to display
    """

    # Get average activation per concept across all patches
    concept_avg_activations = concept_activations.mean(axis=0)

    # Get top K concepts sorted by average activation
    top_concept_indices = np.argsort(concept_avg_activations)[-top_k:][::-1]

    fig, axes = plt.subplots(top_k, 3, figsize=(5, 3*top_k))

    if axes.ndim == 1:
        axes = axes.reshape(-1, 3)

    for k, concept_idx in enumerate(top_concept_indices):
        # Get activation scores for this concept
        concept_scores = concept_activations[:, concept_idx]

        # Find top 3 crops with highest activation
        top_3_indices = np.argsort(concept_scores)[-3:][::-1]
        top_3_scores = concept_scores[top_3_indices]

        # Display top 3 crops
        for col, (crop_idx, score) in enumerate(zip(top_3_indices, top_3_scores)):
            ax = axes[k, col]
            crop = crops[crop_idx]

            # Handle both grayscale and RGB
            if crop.ndim == 3 and crop.shape[2] == 1:
                ax.imshow(crop.squeeze(), cmap='gray', vmin=0, vmax=1)
            elif crop.ndim == 2:
                ax.imshow(crop, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(crop)

            ax.set_title(f"Concept {concept_idx}\nCrop #{crop_idx}\nActivation: {score:.3f}",
                        fontsize=10, fontweight='bold' if col == 0 else 'normal')

            if col == 0:
                # Keep spines visible for the red border, turn off ticks/labels
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.show()
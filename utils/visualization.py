import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import  Subset
from core.cbdt_layers import visualize_decision_journey


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
"""
Concept-Based Decision Tree Classifier
Visualizes decision paths through concept activations with top patches at each node
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ConceptBasedDecisionTree:
    """
    A Decision Tree Classifier that operates on concept activations
    and visualizes decisions with concept patches.
    
    This version works with CRAFT outputs where:
    - crops: actual patch images [n_patches, patch_h, patch_w, channels]
    - concept_activations: activations for each patch [n_patches, n_concepts]
    """
    
    def __init__(self, crops: np.ndarray, 
                 concept_activations: np.ndarray,
                 max_depth: int = 5, 
                 min_samples_split: int = 5,
                 random_state: int = 42):
        """
        Args:
            crops: Array of shape [n_patches, patch_h, patch_w, channels]
                   Actual patch images from CRAFT
            concept_activations: Array of shape [n_patches, n_concepts]
                               Concept activations for each patch
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            random_state: Random seed
        """
        self.crops = crops
        self.concept_activations = concept_activations
        self.n_concepts = concept_activations.shape[1]
        self.patch_h, self.patch_w = crops.shape[1], crops.shape[2]
        self.channels = crops.shape[3] if crops.ndim == 4 else 1
        
        # Extract top patch for each concept
        self.concept_patches = self._extract_concept_patches()
        
        # Initialize decision tree
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.is_fitted = False
        self.feature_names = [f"Concept_{i}" for i in range(self.n_concepts)]
        self.class_names = [str(i) for i in range(10)]  # For MNIST 0-9
    
    def _extract_concept_patches(self) -> np.ndarray:
        """
        Extract the top patch for each concept based on concept activations.
        
        Returns:
            Array of shape [n_concepts, patch_h, patch_w] (grayscale)
        """
        concept_patches = []
        
        for concept_id in range(self.n_concepts):
            # Find patch with highest activation for this concept
            activations = self.concept_activations[:, concept_id]
            top_patch_idx = np.argmax(activations)
            
            # Get the patch
            top_patch = self.crops[top_patch_idx]  # [patch_h, patch_w, channels]
            
            # Convert to grayscale if needed
            if self.channels == 1:
                top_patch = top_patch.squeeze()  # [patch_h, patch_w]
            elif self.channels == 3:
                # Convert RGB to grayscale
                top_patch = np.mean(top_patch, axis=2)  # [patch_h, patch_w]
            
            concept_patches.append(top_patch)
        
        return np.array(concept_patches)  # [n_concepts, patch_h, patch_w]
        
    def train(self, concept_activations: np.ndarray, labels: np.ndarray):
        """
        Train the decision tree on concept activations.
        
        Args:
            concept_activations: Array of shape [n_samples, n_concepts]
            labels: Array of shape [n_samples,]
        """
        print("Training Decision Tree on Concept Activations...")
        print(f"  Shape: {concept_activations.shape}")
        print(f"  Classes: {np.unique(labels)}")
        
        self.tree.fit(concept_activations, labels)
        self.is_fitted = True
        
        train_acc = self.tree.score(concept_activations, labels)
        print(f"  Training Accuracy: {train_acc*100:.2f}%")
        
    def evaluate(self, concept_activations: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the tree on test data.
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.is_fitted:
            raise ValueError("Tree must be trained first!")
            
        predictions = self.tree.predict(concept_activations)
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'predictions': predictions,
            'confusion_matrix': confusion_matrix(labels, predictions)
        }
    
    def get_decision_path(self, concept_activation: np.ndarray) -> Tuple[List[int], List[float], int]:
        """
        Get the decision path taken by the tree for a single sample.
        
        Args:
            concept_activation: Array of shape [n_concepts,]
            
        Returns:
            (node_indices, feature_values_at_nodes, predicted_class)
        """
        if not self.is_fitted:
            raise ValueError("Tree must be trained first!")
        
        # Get decision path
        decision_path = self.tree.decision_path([concept_activation]).indices[0]
        
        # Get the values of concepts at each decision node
        feature_values = []
        for node_idx in decision_path:
            feature_idx = self.tree.tree_.feature[node_idx]
            if feature_idx >= 0:  # Not a leaf node
                feature_values.append(concept_activation[feature_idx])
            else:
                feature_values.append(None)
        
        predicted_class = self.tree.predict([concept_activation])[0]
        
        return list(decision_path), feature_values, predicted_class
    
    def predict(self, concept_activations: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.tree.predict(concept_activations)
    
    def get_feature_at_node(self, node_idx: int) -> int:
        """Get the feature (concept) used for splitting at a node."""
        return self.tree.tree_.feature[node_idx]
    
    def get_threshold_at_node(self, node_idx: int) -> float:
        """Get the threshold value for splitting at a node."""
        return self.tree.tree_.threshold[node_idx]
    
    def get_node_class(self, node_idx: int) -> int:
        """Get the most common class at a node."""
        value = self.tree.tree_.value[node_idx][0]
        return np.argmax(value)
    
    def visualize_tree_structure(self, figsize: Tuple[int, int] = (20, 12)):
        """
        Visualize the tree structure with concept patches at each node.
        """
        if not self.is_fitted:
            raise ValueError("Tree must be trained first!")
        
        fig, ax = plt.subplots(figsize=figsize)
        self._draw_tree_recursive(0, ax, x=0.5, y=1.0, dx=0.25)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        plt.title("Decision Tree with Concept Patches", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _draw_tree_recursive(self, node_idx: int, ax, x: float, y: float, dx: float):
        """Recursively draw tree nodes with patches."""
        
        # Draw node box with patch
        feature_idx = self.tree.tree_.feature[node_idx]
        
        if feature_idx >= 0:  # Not a leaf
            # Decision node
            concept_patch = self.concept_patches[feature_idx]
            threshold = self.tree.tree_.threshold[node_idx]
            
            # Draw patch as node
            patch_img = torch.from_numpy(concept_patch).float()
            if patch_img.dim() == 2:  # Grayscale
                patch_img = patch_img.unsqueeze(0)
            
            # Add patch to plot at position (x, y)
            ax_inset = ax.inset_axes([x-0.04, y-0.08, 0.08, 0.08])
            ax_inset.imshow(patch_img[0].numpy() if patch_img.shape[0] > 1 else patch_img.squeeze().numpy(), 
                           cmap='gray')
            ax_inset.set_title(f"C{feature_idx}\n<{threshold:.2f}", fontsize=8)
            ax_inset.axis('off')
            
            # Draw edges to children
            left_child = self.tree.tree_.children_left[node_idx]
            right_child = self.tree.tree_.children_right[node_idx]
            
            if left_child != -1:
                x_left = x - dx
                y_left = y - 0.2
                ax.arrow(x, y-0.04, x_left-x, y_left-y+0.04, 
                        head_width=0.02, head_length=0.02, fc='black', ec='black', alpha=0.6)
                self._draw_tree_recursive(left_child, ax, x_left, y_left, dx/2)
            
            if right_child != -1:
                x_right = x + dx
                y_right = y - 0.2
                ax.arrow(x, y-0.04, x_right-x, y_right-y+0.04, 
                        head_width=0.02, head_length=0.02, fc='black', ec='black', alpha=0.6)
                self._draw_tree_recursive(right_child, ax, x_right, y_right, dx/2)
        else:
            # Leaf node
            class_pred = self.get_node_class(node_idx)
            rect = FancyBboxPatch((x-0.04, y-0.04), 0.08, 0.08, 
                                 boxstyle="round,pad=0.01", 
                                 edgecolor='green', facecolor='lightgreen', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, str(class_pred), ha='center', va='center', 
                   fontsize=10, fontweight='bold')


def visualize_decision_journey(
    image: torch.Tensor,
    concept_activation: np.ndarray,
    concept_patches: np.ndarray,
    tree: ConceptBasedDecisionTree,
    true_label: int,
    predicted_label: int,
    image_index: int = None
):
    """
    Visualize the decision journey for a single image through the tree.
    
    Args:
        image: Input image tensor [1, H, W] or [3, H, W]
        concept_activation: Concept activation vector [n_concepts,]
        concept_patches: Array of concept patches
        tree: Trained ConceptBasedDecisionTree
        true_label: Ground truth label
        predicted_label: Model's predicted label
        image_index: Optional index for display
    """
    
    # Get decision path
    node_path, feature_values, tree_prediction = tree.get_decision_path(concept_activation)
    
    # Prepare figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # ============ Top Left: Original Image ============
    ax_img = fig.add_subplot(gs[0, 0])
    img_display = image.permute(1, 2, 0).numpy() if image.dim() == 3 else image.numpy()
    if img_display.shape[-1] == 1 or image.shape[0] == 1:
        ax_img.imshow(img_display.squeeze(), cmap='gray')
    else:
        ax_img.imshow(img_display)
    ax_img.set_title(f"Input Image\nIdx: {image_index if image_index else '?'}", fontweight='bold')
    ax_img.axis('off')
    
    # ============ Top Middle: Prediction Info ============
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_pred.axis('off')
    correct = "✓" if true_label == predicted_label else "✗"
    pred_text = f"""
    PREDICTION SUMMARY
    ─────────────────────
    True Label: {true_label}
    Predicted: {predicted_label} {correct}
    Confidence: {concept_activation.max():.3f}
    Tree Depth: {len(node_path)}
    """
    ax_pred.text(0.1, 0.5, pred_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============ Top Right: Concept Activations ============
    ax_concepts = fig.add_subplot(gs[0, 2:])
    sorted_indices = np.argsort(concept_activation)[::-1]
    top_k = 8
    ax_concepts.barh(range(top_k), concept_activation[sorted_indices[:top_k]])
    ax_concepts.set_yticks(range(top_k))
    ax_concepts.set_yticklabels([f"C{idx}" for idx in sorted_indices[:top_k]])
    ax_concepts.set_xlabel("Activation Value")
    ax_concepts.set_title("Top Concept Activations", fontweight='bold')
    ax_concepts.invert_yaxis()
    
    # ============ Middle: Decision Path with Patches ============
    n_nodes = len(node_path)
    cols_per_node = max(1, 4 // min(n_nodes, 4))
    
    for i, node_idx in enumerate(node_path):
        ax = fig.add_subplot(gs[1, i % 4])
        
        feature_idx = tree.get_feature_at_node(node_idx)
        
        if feature_idx >= 0:  # Decision node
            concept_patch = concept_patches[feature_idx]
            threshold = tree.get_threshold_at_node(node_idx)
            activation = concept_activation[feature_idx]
            
            # Display patch
            if isinstance(concept_patch, np.ndarray):
                patch_np = concept_patch
            else:
                patch_np = concept_patch.cpu().numpy() if hasattr(concept_patch, 'cpu') else concept_patch
            
            ax.imshow(patch_np, cmap='gray')
            
            # Decision direction
            direction = "LEFT" if activation < threshold else "RIGHT"
            color = 'lightcoral' if direction == "LEFT" else 'lightgreen'
            
            ax.set_title(f"Node {node_idx}\nC{feature_idx} = {activation:.3f}\nThreshold: {threshold:.3f}\n→ {direction}",
                        fontweight='bold', fontsize=9, bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
            ax.axis('off')
        else:
            # Leaf node
            class_at_leaf = tree.get_node_class(node_idx)
            ax.text(0.5, 0.5, f"LEAF\nClass: {class_at_leaf}", 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax.set_title(f"Node {node_idx} (Leaf)", fontweight='bold')
            ax.axis('off')
    
    # ============ Bottom: Decision Path Summary ============
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    path_text = "DECISION PATH:\n"
    path_text += "─" * 80 + "\n"
    
    for i, node_idx in enumerate(node_path):
        feature_idx = tree.get_feature_at_node(node_idx)
        if feature_idx >= 0:
            activation = concept_activation[feature_idx]
            threshold = tree.get_threshold_at_node(node_idx)
            direction = "LEFT (<)" if activation < threshold else "RIGHT (≥)"
            path_text += f"Step {i}: Node {node_idx} → Concept {feature_idx} | "
            path_text += f"Activation: {activation:.3f} | Threshold: {threshold:.3f} | {direction}\n"
        else:
            class_at_leaf = tree.get_node_class(node_idx)
            path_text += f"Step {i}: Node {node_idx} (LEAF) → Predicted Class: {class_at_leaf}\n"
    
    ax_summary.text(0.05, 0.95, path_text, fontsize=9, family='monospace',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle("Decision Tree Journey Through Concepts", fontsize=16, fontweight='bold', y=0.995)
    plt.show()


def create_confusion_matrix_visualization(predictions: np.ndarray, labels: np.ndarray):
    """
    Create a confusion matrix heatmap.
    """
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()


# ============ USAGE EXAMPLE ============

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     Concept-Based Decision Tree with Visualization            ║
    ║                                                                ║
    ║  This module provides:                                         ║
    ║  1. ConceptBasedDecisionTree - Tree operating on concepts     ║
    ║  2. visualize_decision_journey() - Path visualization         ║
    ║  3. create_confusion_matrix_visualization() - Metrics         ║
    ╚════════════════════════════════════════════════════════════════╝
    
    INTEGRATION WITH YOUR CODE:
    ───────────────────────────
    
    After computing concept activations from CRAFT:
    
        # Assume you have:
        # - crops_u: [n_patches, patch_dim]
        # - W: [patch_dim, n_concepts] from NMF
        # - train_labels, test_labels
        # - concept_patches: top patch for each concept
        
        # 1. Compute activations
        concept_activations = crops_u @ W  # [n_patches, n_concepts]
        
        # 2. Create and train tree
        tree = ConceptBasedDecisionTree(
            concept_patches=concept_patches,
            max_depth=5,
            min_samples_split=5
        )
        tree.train(concept_activations[:train_size], train_labels)
        
        # 3. Evaluate
        metrics = tree.evaluate(concept_activations[train_size:], test_labels)
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        
        # 4. Visualize tree structure
        tree.visualize_tree_structure()
        
        # 5. Visualize decision journey for a specific image
        test_image = test_ds[0][0]  # First test image
        test_activation = concept_activations[train_size]
        true_label = test_labels[0]
        pred_label = tree.predict(test_activation.reshape(1, -1))[0]
        
        visualize_decision_journey(
            image=test_image,
            concept_activation=test_activation,
            concept_patches=concept_patches,
            tree=tree,
            true_label=true_label,
            predicted_label=pred_label,
            image_index=0
        )
    """)
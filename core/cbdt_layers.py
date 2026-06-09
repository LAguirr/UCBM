import os
from typing import Union, Literal, Tuple, List, Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder


class ConceptBasedDecisionTree:
  def __init__(self,
               backbone,
               h: Union [torch.Tensor, np.ndarray],
               crops: np.ndarray,
               concept_activations: np.ndarray | torch.Tensor,
               max_depth: int,
               min_samples_split: int,
               batch_size: int,
               device: Literal['cuda', 'cpu'] ):


        self._backbone = backbone # we have it, FeatureExtractor
        if not torch.is_tensor(h):
            h = torch.tensor(h) # h is the concepts from h = np.load("craft_concept_bank.npy")
        self.n_concepts = h.shape[0]  #shape (10, 64)
        self._h = h.to(device)
        self._h = self._h / torch.norm(self._h, dim=1, keepdim=True) #Normalize the concept


        self.crops = crops
        self.concept_activations = concept_activations
        self._batch_size = batch_size #we have it from the DataLoader = 64
        self._device = device #GPU. Thanks google

        print("Crops shape", crops.shape)
        self.patch_h, self.patch_w = crops.shape[0], crops.shape[1]
        self.channels = crops.shape[3] if crops.ndim == 4 else 1

        self.top_concept_patches = self._extract_concept_patches()
        print("Concept patches shape: ", self.top_concept_patches.shape)

        # Initialize decision tree
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
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
        top_concept_patches = []

        for concept_id in range(self.n_concepts):
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

            top_concept_patches.append(top_patch)

        return np.array(top_concept_patches)  # [n_concepts, patch_h, patch_w]


  @torch.no_grad()
  def _get_concept_embeddings(self,dataset: Dataset,
                                saved_activation_path: Optional[str] = None,
                                data_label: Optional[str] = None,
                                normalize=False) \
                                    -> Dataset[torch.Tensor]:


        return raw_concept_sims(self._h,dataset,
                                self._backbone,self._batch_size,
                                self._device,saved_activation_path,
                                data_label)


  def train(self, data_ds: ImageFolder):
        """
        Train the decision tree on concept activations.
        Args:
            data_ds: Array of shape [n_samples, n_concepts]

        """
        embeddings = self._get_concept_embeddings(data_ds, act_path, "train")

        num_embeddings = len(embeddings)

        # --- DataLoader over (embeddings, targets) pairs ---
        dset = PDataset(embeddings, data_ds.targets[:num_embeddings])
        data_loader = DataLoader(dset, self._batch_size, shuffle=True, num_workers=num_workers)

        all_gated = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(data_loader, leave=False):
                X_batch = X_batch.to(self._device)
                X_batch = (X_batch - X_batch.mean(dim=0)) / (X_batch.std(dim=0) + 1e-8)
                after_gate = X_batch.detach()


                all_gated.append(after_gate.cpu())
                all_labels.append(y_batch.cpu())

        all_gated = torch.cat(all_gated, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        print("all_gated: ", all_gated.shape)
        print("all_labels: ", all_labels.shape)


        print("Training Decision Tree on Concept Activations...")
        print(f"Embeddings Shape: {embeddings.shape}")
        print(f"  Classes: {np.unique(labels)}")
        self.tree.fit(all_gated, all_labels)
        self.is_fitted = True
        train_acc = self.tree.score(all_gated, all_labels)
        print(f"\nTraining Accuracy: {train_acc*100:.2f}%")


        return {
            'accuracy': train_acc
        }


  def evaluate(self, data_ds: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the tree on test data.

        Returns:
            Dictionary with accuracy metrics
        """
        if not self.is_fitted:
            raise ValueError("Tree must be trained first!")

        embeddings = self._get_concept_embeddings(
            data_ds,
            act_path,
            "test"
        )

        num_embeddings = len(embeddings)
        dset = PDataset(embeddings, data_ds.targets[:num_embeddings])
        data_loader = DataLoader(dset, self._batch_size, shuffle=True, num_workers=num_workers)

        all_gated = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(data_loader, leave=False):
                X_batch = X_batch.to(self._device)
                X_batch = (X_batch - X_batch.mean(dim=0)) / (X_batch.std(dim=0) + 1e-8)
                after_gate = X_batch.detach()
                all_gated.append(after_gate.cpu())
                all_labels.append(y_batch.cpu())

        all_gated = torch.cat(all_gated, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        print("all_gated: ", all_gated.shape)
        print("all_labels: ", all_labels.shape)

        predictions = self.tree.predict(all_gated)
        acc = accuracy_score(all_labels, predictions)
        print(f"Test Accuracy: {100 * acc:.2f}%")


        return {
            'accuracy': acc,
            'predictions': predictions,
            'confusion_matrix': confusion_matrix(all_labels, predictions)
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
        decision_path = self.tree.decision_path([concept_activation]).indices

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


  def visualize_tree_structure(self, figsize: Tuple[int, int] = (20, 14)):
    """
    Visualize the tree structure with concept patches at each node.
    Root node is largest; each deeper level shrinks proportionally down to a minimum.
    """
    if not self.is_fitted:
        raise ValueError("Tree must be trained first!")

    max_depth = self.tree.get_depth()
    INITIAL_DX = 0.44

    # Min node size: mathematically prevents horizontal overlap at deepest level
    MIN_NODE = min(0.055, (INITIAL_DX / (2 ** (max_depth - 1))) * 0.82) if max_depth > 0 else 0.055
    # Max node size: root is up to 4x the minimum, capped at 0.13
    MAX_NODE = min(0.13, max(MIN_NODE * 4.0, MIN_NODE + 0.05))
    # Fixed level gap: evenly fills the figure vertically
    LEVEL_GAP = 0.85 / max(1, max_depth)

    def node_size_at(depth: int) -> float:
        """Linear taper from MAX_NODE at root down to MIN_NODE at max_depth."""
        if max_depth == 0:
            return MAX_NODE
        t = depth / max_depth
        return MAX_NODE * (1.0 - t) + MIN_NODE * t

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('#f9f9f9')

    self._draw_tree_recursive(
        0, ax, x=0.5, y=0.95, dx=INITIAL_DX,
        depth=0, node_size_fn=node_size_at, level_gap=LEVEL_GAP
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


  def _draw_tree_recursive(
    self, node_idx: int, ax,
    x: float, y: float, dx: float,
    depth: int, node_size_fn, level_gap: float
  ):
    """Recursively draw tree nodes; node size tapers with depth."""

    feature_idx = self.tree.tree_.feature[node_idx]
    node_size = node_size_fn(depth)
    half = node_size / 2
    font_size = max(4.5, min(9.0, node_size * 95))

    if feature_idx >= 0:  # Decision node
        concept_patch = self.top_concept_patches[feature_idx]
        threshold = self.tree.tree_.threshold[node_idx]

        patch_img = torch.from_numpy(concept_patch).float()
        if patch_img.dim() == 2:
            patch_img = patch_img.unsqueeze(0)

        img_data = patch_img.numpy()
        if img_data.ndim == 1:
            img_data = np.expand_dims(img_data, axis=0)
        elif img_data.ndim >= 3 and img_data.shape[0] > 1:
            img_data = img_data[0]
        elif img_data.ndim == 3 and img_data.shape[0] in [1, 3, 4]:
            img_data = np.transpose(img_data, (1, 2, 0))

        ax_inset = ax.inset_axes([x - half, y - node_size, node_size, node_size])
        ax_inset.imshow(img_data, cmap='gray', aspect='auto')
        ax_inset.set_title(
            f"C{feature_idx}\n<{threshold:.2f}",
            fontsize=font_size, pad=1.5,
            color='#222222', fontweight='bold'
        )
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(max(0.4, node_size * 12))
        ax_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        left_child  = self.tree.tree_.children_left[node_idx]
        right_child = self.tree.tree_.children_right[node_idx]

        child_size = node_size_fn(depth + 1)
        arrow_props = dict(
            arrowstyle='-|>',
            color='#888888',
            lw=max(0.5, node_size * 10),
            mutation_scale=max(5, node_size * 90),
        )

        if left_child != -1:
            x_left  = x - dx / 2
            y_left  = y - level_gap
            # Arrow starts at bottom-center of parent, ends at top-center of child
            ax.annotate(
                '',
                xy=(x_left, y_left - child_size + child_size),
                xytext=(x, y - node_size),
                arrowprops=arrow_props
            )
            self._draw_tree_recursive(
                left_child, ax, x_left, y_left, dx / 2,
                depth + 1, node_size_fn, level_gap
            )

        if right_child != -1:
            x_right = x + dx / 2
            y_right = y - level_gap
            ax.annotate(
                '',
                xy=(x_right, y_right),
                xytext=(x, y - node_size),
                arrowprops=arrow_props
            )
            self._draw_tree_recursive(
                right_child, ax, x_right, y_right, dx / 2,
                depth + 1, node_size_fn, level_gap
            )

    else:  # Leaf node
        class_pred = self.get_node_class(node_idx)
        rect = FancyBboxPatch(
            (x - half, y - node_size), node_size, node_size,
            boxstyle="round,pad=0.004",
            edgecolor='#2e7d32', facecolor='#a5d6a7',
            linewidth=max(0.6, node_size * 15), zorder=3
        )
        ax.add_patch(rect)
        ax.text(
            x, y - node_size / 2, str(class_pred),
            ha='center', va='center',
            fontsize=max(5, font_size * 1.1),
            fontweight='bold', color='#1b5e20', zorder=4
        )

  @torch.no_grad()
  def get_evaluation_metric(self,
                              dataset: ImageFolder,
                              metric: list[Literal["acc", "auprc", "auroc", "auprc_pc"]] = ["acc"],
                              saved_activation_path: Optional[str] = None,
                              data_label: Optional[str] = None) \
                                -> dict[str, float]:


        if isinstance(dataset, Subset):
            parent = dataset.dataset
            indices = dataset.indices
        else:
            parent = dataset
            indices = None

        if isinstance(parent, ConcatDataset):
            all_targets = torch.cat([d.targets for d in parent.datasets])
        else:
            all_targets = parent.targets

        if indices is not None:
            current_targets = all_targets[indices]
        else:
            current_targets = all_targets

        # Load the concept activations.
        embeddings = self._get_concept_embeddings(dataset, saved_activation_path)


        dset = TensorDataset(embeddings.cpu(), current_targets.cpu())
        data_loader = DataLoader(dset, batch_size=self._batch_size, shuffle=False, num_workers=4)

        y_pred = []
        y_true = []

        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(self._device)
            X_batch = (X_batch - X_batch.mean(dim=0)) / (X_batch.std(dim=0) + 1e-8)
            after_gate = X_batch.detach()

            y_pred.append(after_gate.cpu())
            y_true.append(y_batch.cpu())


        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        preds = self.tree.predict(y_pred)
        acc = accuracy_score(y_true, preds)

        if "acc" in metric:
            return {"acc": acc}

  def save_to_file(self, filepath: str, filename: str):
     def get_backbone():
        try:
            return self._backbone.cpu()
        except AttributeError:
            return None

        path = os.path.join(filepath, filename)
        torch.save(
            {
                "backbone": get_backbone(),
                "batch_size": self._batch_size,
                "num_concepts": self._num_concepts,
                "w": self._h.detach().cpu(),
            }, path)


  def get_info_dict(self,
                      training_data: ImageFolder,
                      test_data: ImageFolder,
                      val_data: ImageFolder,
                      act_bank_path: str,
                      images_preprocessed: int,
                      patch_size: int,
                      total_patches: int,
                      metrics = ["acc", "auprc", "auprc_pc", "auroc"]) -> dict:


        data = dict()
        data["amount of concepts"] = int(self._h.shape[0])
        data["amount of classes"] = len(test_data.classes)
        data["amount of samples"] = images_preprocessed
        data["samples per class"] = images_preprocessed//len(test_data.classes)
        data["total patches"] = total_patches
        data["patch_size"] = int(patch_size)
        first_sample_features, first_sample_label = training_data[0]
        print("\ntraining_data: ", first_sample_features.shape, first_sample_label)
        print("test_data: ", len(test_data))
        print("val_data: ", len(val_data))
        train_res = self.get_evaluation_metric(training_data, metrics, act_bank_path, "train")
        test_res = self.get_evaluation_metric(test_data, metrics, act_bank_path, "test")
        val_res = self.get_evaluation_metric(val_data, metrics, act_bank_path, "val")
        print("train_res: ", train_res)
        print("test_res: ", test_res)
        print("val_res: ", val_res)
        print("min_samples_split: ", self.tree.min_samples_split)
        print("max_depth: ", self.tree.max_depth)

        data['min_samples_split'] = self.tree.min_samples_split
        data['max_depth'] = self.tree.max_depth

        if "acc" in metrics and "acc" in train_res:
            data["train acc"] = train_res["acc"]
            data["test acc"] = test_res["acc"]

        data["val_acc"] = val_res['acc']

        return data

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
    node_path = list(tree.tree.decision_path([concept_activation]).indices)
    feature_values, tree_prediction = None, None

    # Prepare figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)

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
        row_idx = 1 + (i // 4)  # Starts at row 1, moves to row 2 after 4 items, etc.
        col_idx = i % 4

        ax = fig.add_subplot(gs[row_idx, col_idx])

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
                        fontweight='bold', fontsize=8, bbox=dict(boxstyle='round', facecolor=color, alpha=0.6))
            ax.axis('off')
        else:
            # Leaf node
            class_at_leaf = tree.get_node_class(node_idx)
            ax.set_title(f"Node {node_idx} (Leaf)", fontweight='bold')
            ax.text(0.5, 0.5, f"LEAF\nClass: {class_at_leaf}",
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

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

    ax_summary.text(0.48,1.0, path_text, fontsize=9, family='monospace',
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))

    plt.suptitle("Decision Tree Journey Through Concepts", fontsize=16, fontweight='bold', y=0.995)
    plt.show()

# Unsupervised Concept-Based Decision Tree Classifier

This repository implements the Unsupervised Concept-Based Decision Tree Classifier, an interpretable and explainable classification pipeline. It integrates deep neural representations with unsupervised concept extraction (using CRAFT) to build transparent decision structures.

Unlike traditional black-box deep learning models, this architecture allows visual inspection of why a specific classification decision is reached by tracking the activation of human-interpretable concepts at each node of a decision tree.



---

## Architecture Overview

The system consists of three main stages:

1. **Backbone CNN**: A base model trained on the input dataset (e.g., MNIST) to learn strong feature extractors. The features are extracted after convolutional blocks using a global average pooling layer.
2. **Concept Discovery (CRAFT)**: Analyzes spatial convolutional activations to discover localized concept vectors in an unsupervised manner.
3. **Concept-Based Decision Tree (CBDT)**: A custom decision tree trained on concept activations, where each node splits based on the presence of a specific visual concept.

---

## File Structure

- [main.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/main.py): Pipeline script that initializes the dataset, trains the CNN backbone (if not cached), performs concept extraction via CRAFT, trains the decision tree classifier, and triggers path visualizations.
- [core/backbone.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/backbone.py): Neural network definition including [Net](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/backbone.py#L14), [FeatureExtractorG](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/backbone.py#L30), and [ClassifierH](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/backbone.py#L43), as well as standard training and validation loops.
- [core/cbdt_layers.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/cbdt_layers.py): Core logic for [ConceptBasedDecisionTree](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/cbdt_layers.py#L21). It manages training, scoring, decision path computation, and complex visualizations.
- [core/dataset_utils.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/dataset_utils.py): Utilities for loading data, generating stratified splits, and preprocessing images for concept discovery.
- [mycraft/craft_torch.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/mycraft/craft_torch.py): PyTorch implementation of the Concept Recursive Activation Feature Tomography (CRAFT) algorithm for unsupervised concept discovery.
- [utils/concept_ops.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/utils/concept_ops.py): Mathematical modules and functions for concept similarity calculations, including JumpReLU, StepFunction, and TopK modules.
- [utils/visualization.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/utils/visualization.py): Custom visualization helpers for analyzing digit classifications, concept hotspots, concept patches, and decision journeys.
- [requirements.txt](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/requirements.txt): Python dependency file.

---

## Installation

Ensure that you have Python installed, then clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Note that the requirements file includes a direct Git reference to the CRAFT implementation library:
`git+https://github.com/LAguirr/Craft.git`

---

## Usage

### Training the Decision Tree Pipeline
To train the backbone model, run CRAFT concept discovery, build the decision tree, and generate decision journey visualizations:

```bash
python main.py
```

Upon successful execution, the trained tree will be saved to `concept_decision_tree.pkl` and metadata will be logged to `Model/info.json`.

---

## Explainability and Visualizations

This project provides several built-in tools for interpretability:

- **Decision Journeys**: Traces the activation values and decision direction (Left/Right) of concepts for individual test images. See the `visualize_decision_journey` function in [core/cbdt_layers.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/cbdt_layers.py#L516).
- **Tree Visualization**: Generates a tree plot displaying decision nodes as the visual patches of the splitting concepts. See the `visualize_tree_structure` method in [core/cbdt_layers.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/cbdt_layers.py#L251).
- **Concept Hotspots and Crops**: Identifies visual areas of the input image corresponding to specific concepts and shows the highest-activating concept crops. See functions in [utils/visualization.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/utils/visualization.py).

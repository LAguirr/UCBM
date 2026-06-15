# Unsupervised Concept-Based Decision Tree Classifier

This repository implements the Unsupervised Concept-Based Decision Tree Classifier, an interpretable and explainable classification pipeline. It integrates deep neural representations with unsupervised concept extraction (using CRAFT and NMF) to build transparent decision structures.

Unlike traditional black-box deep learning models, this architecture allows visual inspection of why a specific classification decision is reached by tracking the activation of human-interpretable concepts at each node of a decision tree.

---

## Architecture Overview

The system consists of three main stages:

1. **Backbone CNN**: A base model trained on the input dataset (e.g., MNIST) to learn strong feature extractors. The features are extracted after convolutional blocks using a global average pooling layer.
2. **Concept Discovery**:
   - **CRAFT (Concept Recursive Activation Feature Tomography)**: Analyzes spatial convolutional activations to discover localized concept vectors.
   - **NMF (Non-negative Matrix Factorization)**: Factorizes spatial activation tensors into concept matrices, ensuring non-negativity and alignment with the projection space.
3. **Concept-Based Decision Classifier**:
   - **Concept-Based Decision Tree (CBDT)**: A custom decision tree trained on concept activations, where each node splits based on the presence of a specific visual concept.
   - **Gated UCBM Classifier**: A three-phase gated linear model where learning offsets are used to regularize and sparse-select concepts during prediction.

---

## File Structure

- [main.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/main.py): Pipeline script that initializes the dataset, trains the CNN backbone (if not cached), performs concept extraction via CRAFT, trains the decision tree classifier, and triggers path visualizations.
- [predict.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/predict.py): Self-contained script implementing the three-phase gated UCBM classifier. It details the NMF factorization, cosine projections, and three-phase training (dense, open gate, and regularized gate).
- [core/backbone.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/backbone.py): Neural network definition including [Net](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/backbone.py#L14), [FeatureExtractorG](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/backbone.py#L30), and [ClassifierH](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/backbone.py#L43), as well as standard training and validation loops.
- [core/cbdt_layers.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/cbdt_layers.py): Core logic for [ConceptBasedDecisionTree](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/cbdt_layers.py#L21). It manages training, scoring, decision path computation, and complex visualizations.
- [core/dataset_utils.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/dataset_utils.py): Utilities for loading data, generating stratified splits, and preprocessing images for concept discovery.
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

### 1. Training the Decision Tree Pipeline
To train the backbone model, run CRAFT concept discovery, build the decision tree, and generate decision journey visualizations:

```bash
python main.py
```

Upon successful execution, the trained tree will be saved to `concept_decision_tree.pkl` and metadata will be logged to `Model/info.json`.

### 2. Running the Gated UCBM Pipeline
To execute the three-phase gated classifier training and generate visualization figures:

```bash
python predict.py
```

This script trains the model in three distinct stages:
- **Phase 1 (Dense)**: Trains a linear classifier directly on concept projections (matches probe accuracy).
- **Phase 2 (Identity Gate)**: Enables the gating parameter with an offset initialized to 0.0 to ensure a stable baseline.
- **Phase 3 (Sparse Fine-Tune)**: Applies regularisation to push the offsets up, introducing sparsity while preserving accuracy.

---

## Explainability and Visualizations

This project provides several built-in tools for interpretability:

- **Concept Exemplar Patches**: Displays the top image crops that activate specific discovered concepts. Saved under `outputs/ucbm_craft_concepts.png` when running `predict.py`.
- **Decision Journeys**: Traces the activation values and decision direction (Left/Right) of concepts for individual test images. See the `visualize_decision_journey` function in [core/cbdt_layers.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/cbdt_layers.py#L516).
- **Tree Visualization**: Generates a tree plot displaying decision nodes as the visual patches of the splitting concepts. See the `visualize_tree_structure` method in [core/cbdt_layers.py](file:///c:/Users/lino_/OneDrive/Documentos/Master1DataIA/NeuroSymbolicAI/UCBM/core/cbdt_layers.py#L251).
- **Decision Contributions**: Plots bar charts showing the positive or negative contribution of each concept to the final prediction. Saved under `outputs/ucbm_craft_decisions.png` when running `predict.py`.

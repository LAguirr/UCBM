"""Notebook-style concept discovery helpers for the MNIST tree pipeline.

The notebook builds a concept bank by sampling spatial feature maps from the
backbone, flattening each 7x7 location into a 64-d descriptor, and fitting NMF.
This module makes that workflow reusable from the Python project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import NMF


@dataclass
class ConceptDiscoveryResult:
    """Outputs from the notebook-style concept discovery pipeline."""

    concept_bank: torch.Tensor
    concept_coefficients: np.ndarray
    sampled_indices: np.ndarray
    spatial_maps: list[np.ndarray]
    raw_images: list[np.ndarray]
    exemplars: dict[int, list[tuple[np.ndarray, int, int, int, int]]]


@torch.no_grad()
def collect_spatial_descriptors(
    dataset,
    backbone: torch.nn.Module,
    sample_size: int,
    device: torch.device | str,
    image_decoder: Callable[[torch.Tensor], np.ndarray] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Sample images and collect spatial descriptors from ``backbone``.

    Returns a matrix shaped ``(sample_size * H * W, channels)`` that can be
    fed directly into NMF, plus the per-image feature maps, raw images, and the
    sampled indices used to build the matrix.
    """

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(dataset), size=sample_size, replace=False)

    spatial_maps: list[np.ndarray] = []
    raw_images: list[np.ndarray] = []
    descriptor_rows: list[np.ndarray] = []

    backbone.eval()
    for idx in sampled_indices:
        image, _ = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        feat = backbone(image_batch).squeeze(0).detach().cpu().numpy()

        if feat.ndim != 3:
            raise ValueError(
                "Backbone must return a 3D feature map per image for concept discovery."
            )

        # feat: (channels, height, width) -> (height * width, channels)
        descriptor_rows.append(feat.reshape(feat.shape[0], -1).T)
        spatial_maps.append(feat)

        if image_decoder is not None:
            raw_images.append(image_decoder(image))
        else:
            raw_images.append(image.squeeze().detach().cpu().numpy())

    nmf_input = np.concatenate(descriptor_rows, axis=0).clip(min=0)
    return nmf_input, spatial_maps, raw_images, sampled_indices


def fit_concept_bank(
    nmf_input: np.ndarray,
    n_concepts: int,
    random_state: int = 42,
    max_iter: int = 1000,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Fit NMF and return the raw coefficients, normalized concept bank, and U."""

    reducer = NMF(
        n_components=n_concepts,
        init="nndsvda",
        max_iter=max_iter,
        random_state=random_state,
    )
    concept_coefficients = reducer.fit_transform(nmf_input)
    concept_bank = reducer.components_.T
    concept_bank = concept_bank / (np.linalg.norm(concept_bank, axis=0, keepdims=True) + 1e-8)
    concept_bank_tensor = torch.tensor(concept_bank, dtype=torch.float32)
    return concept_coefficients, concept_bank_tensor, reducer.components_


def build_concept_exemplars(
    concept_coefficients: np.ndarray,
    raw_images: list[np.ndarray],
    n_concepts: int,
    top_k_patches: int = 6,
    feature_map_shape: tuple[int, int] = (7, 7),
    crop_size: int = 8,
    upsample_size: tuple[int, int] = (28, 28),
) -> dict[int, list[tuple[np.ndarray, int, int, int, int]]]:
    """Select exemplar crops for each concept from the NMF coefficient maps."""

    h_feat, w_feat = feature_map_shape
    per_image = concept_coefficients.reshape(len(raw_images), h_feat, w_feat, n_concepts)

    exemplars: dict[int, list[tuple[np.ndarray, int, int, int, int]]] = {}
    for concept_idx in range(n_concepts):
        max_per_img = per_image[:, :, :, concept_idx].max(axis=(1, 2))
        top_images = np.argsort(max_per_img)[::-1][:top_k_patches]
        concept_examples: list[tuple[np.ndarray, int, int, int, int]] = []

        for image_idx in top_images:
            cmap = per_image[image_idx, :, :, concept_idx]
            cmap_up = F.interpolate(
                torch.tensor(cmap).unsqueeze(0).unsqueeze(0),
                size=upsample_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
            row, col = np.unravel_index(cmap_up.argmax(), upsample_size)
            row0 = max(0, row - crop_size // 2)
            col0 = max(0, col - crop_size // 2)
            row1 = min(upsample_size[0], row0 + crop_size)
            col1 = min(upsample_size[1], col0 + crop_size)
            concept_examples.append((raw_images[image_idx], row0, col0, row1 - row0, col1 - col0))

        exemplars[concept_idx] = concept_examples

    return exemplars


def discover_concepts(
    dataset,
    backbone: torch.nn.Module,
    n_concepts: int,
    sample_size: int,
    device: torch.device | str,
    top_k_patches: int = 6,
    seed: int = 42,
    image_decoder: Callable[[torch.Tensor], np.ndarray] | None = None,
) -> ConceptDiscoveryResult:
    """Run the notebook-style spatial NMF discovery pipeline end to end."""

    nmf_input, spatial_maps, raw_images, sampled_indices = collect_spatial_descriptors(
        dataset=dataset,
        backbone=backbone,
        sample_size=sample_size,
        device=device,
        image_decoder=image_decoder,
        seed=seed,
    )
    concept_coefficients, concept_bank, _ = fit_concept_bank(
        nmf_input=nmf_input,
        n_concepts=n_concepts,
        random_state=seed,
    )
    exemplars = build_concept_exemplars(
        concept_coefficients=concept_coefficients,
        raw_images=raw_images,
        n_concepts=n_concepts,
        top_k_patches=top_k_patches,
    )
    return ConceptDiscoveryResult(
        concept_bank=concept_bank,
        concept_coefficients=concept_coefficients,
        sampled_indices=sampled_indices,
        spatial_maps=spatial_maps,
        raw_images=raw_images,
        exemplars=exemplars,
    )

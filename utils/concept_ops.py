import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from os import path
import torch.nn as nn

@torch.no_grad()
def raw_concept_sims(h, dataset, backbone, batch_size, device, saved_activation_path=None, data_label=None, normalize=False, mean=None, std=None):
    save = data_label is not None
    if save:
        assert saved_activation_path is not None, (
            "`saved_activation_path` must be provided when `data_label` is set."
        )
        saving_dir = path.join(saved_activation_path, f"saved_{data_label}_activations")
        os.makedirs(saving_dir, exist_ok=True)

    
    # --- Prepare CAVs ---
    if not torch.is_tensor(h):
        h = torch.tensor(h, dtype=torch.float32)
    h = h.to(device)
    norms = torch.linalg.norm(h, dim=1, keepdim=True)
    h = h / norms.clamp(min=1e-8) 

    if isinstance(backbone, nn.Module):
        backbone.to(device)
        backbone.eval()
        
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_activations = []

    for i, (X_batch, _) in enumerate(tqdm(data_loader, leave=False)):
        out = backbone(X_batch.to(device))
        if out.ndim == 4: out = out.mean(dim=(2, 3))

        out = out / torch.linalg.norm(out, dim=1, keepdim=True).clamp(min=1e-8)
        out = torch.matmul(out.to(h.dtype), h.T).cpu()

        all_activations.append(out)
        if save: torch.save(out, path.join(saving_dir, f"{data_label}_{i}.pt"))

    full_tensor = torch.cat(all_activations)

    return (full_tensor - full_tensor.mean()) / full_tensor.std().clamp(min=1e-5) if normalize else full_tensor
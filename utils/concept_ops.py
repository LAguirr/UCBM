import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from os import path
import torch.nn as nn
from typing import Optional, Callable

################################################################################
#                                                                              #
#                                   JumpReLU                                   #
#                 JumpReLU with straight-through estimator.                    #
#                Following: https://arxiv.org/abs/2407.14435                   #
#                                                                              #
################################################################################

class RectangleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

def rectangle(x: torch.Tensor) -> torch.Tensor:
    return RectangleFunction.apply(x)

class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                threshold: torch.Tensor,
                bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor): # ste
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * output_grad
        rect_value = rectangle((x - threshold) / bandwidth)
        threshold_grad = -(threshold / bandwidth) * rect_value * output_grad
        return x_grad, threshold_grad, None

class JumpReLU(nn.Module):
    def __init__(self,
                 num_concepts: int,
                 threshold_init: Optional[torch.Tensor] = None,
                 bandwidth: float = 1e-3):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(-10*torch.ones(num_concepts,
                                                         requires_grad=True))
        if threshold_init is not None:
            assert threshold_init.numel() == num_concepts, \
                f"Init threshold is of dimension {threshold_init.size()}" + \
                f", but should be of dimension {num_concepts}. "
            self.log_threshold = threshold_init
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _JumpReLU.apply(
            x,
            self.log_threshold.exp(),  # exp ensures positive threshold
            self.bandwidth
            )


################################################################################
#                                                                              #
#                                    L0-Loss                                   #
#           L0-similar function which is not killing the gradient.             #
#                                                                              #
################################################################################

class _StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                threshold: torch.Tensor,
                bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor): # ste
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = torch.zeros_like(x)
        rect_value = rectangle((x - threshold) / bandwidth)
        threshold_grad = -(1.0 / bandwidth) * rect_value * output_grad
        return x_grad, threshold_grad, None

class StepFunction(nn.Module):
    def __init__(self):
        super(StepFunction, self).__init__()

    def forward(self,
                x: torch.Tensor,
                threshold: torch.Tensor,
                bandwidth: float) -> torch.Tensor:
        return _StepFunction.apply(x, threshold, bandwidth)

step_function = StepFunction()
def l0_loss(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
    out = step_function(x, threshold, bandwidth)
    return torch.sum(out, dim=-1).sum()


def l0_approx(x: torch.Tensor, threshold: torch.Tensor, a: int = 20) -> float:
    out = 1 / (1 + torch.exp(-a*(x - threshold)))
    return out.sum().item()

################################################################################
#                                                                              #
#                                 TopK-Module                                  #
#                  Module that keeps only k largest values.                    #
#      Implementation from  https://github.com/openai/sparse_autoencoder
      #
################################################################################

class TopK(nn.Module):
    def __init__(self,
                 k: int,
                 postact_fn: Optional[Callable] = None):
        super().__init__()
        self.k = k
        # Avoid mutable default argument — a single nn.ReLU() instance would be
        # shared across all TopK instances created without an explicit postact_fn
        self.postact_fn = postact_fn if postact_fn is not None else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sorted=False skips the internal sort step — we only need positions,
        # not ranked order, so this reduces topk from O(n log k) to O(n + k)
        topk = torch.topk(x, k=self.k, dim=-1, sorted=False)
        values = self.postact_fn(topk.values)
        return torch.zeros_like(x).scatter_(-1, topk.indices, values)

    def extra_repr(self) -> str:
        # extra_repr integrates cleanly with PyTorch's print/repr system
        # without polluting state_dict with non-tensor data
        return f"k={self.k}, postact_fn={self.postact_fn.__class__.__name__}"

    def get_config(self) -> dict:
        # Separates hyperparameter serialization from tensor state serialization
        return {"k": self.k, "postact_fn": self.postact_fn.__class__.__name__}

    @classmethod
    def from_config(cls, config: dict) -> "TopK":
        postact_cls = ACTIVATIONS_CLASSES.get(config["postact_fn"])
        if postact_cls is None:
            raise ValueError(
                f"Unknown activation '{config['postact_fn']}'. "
                f"Available: {list(ACTIVATIONS_CLASSES.keys())}"
            )
        return cls(k=config["k"], postact_fn=postact_cls())

ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}


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
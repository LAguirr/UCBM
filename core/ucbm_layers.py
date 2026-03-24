import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from tqdm import tqdm, trange
import numpy as np
from utils.concept_ops import raw_concept_sims

from torcheval.metrics.functional import multiclass_accuracy
from typing import Optional, Literal, Union, Callable
from core.dataset_utils import PDataset

# --- Activation and Loss Utils ---
class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * output_grad
        rect_value = (( (x-threshold)/bandwidth > -0.5) & ((x-threshold)/bandwidth < 0.5)).to(x.dtype)
        threshold_grad = -(threshold / bandwidth) * rect_value * output_grad
        return x_grad, threshold_grad, None

class JumpReLU(nn.Module):
    def __init__(self, num_concepts, threshold_init=None, bandwidth=1e-3):
        super().__init__()
        self.log_threshold = nn.Parameter(-10*torch.ones(num_concepts, requires_grad=True))
        self.bandwidth = bandwidth
    def forward(self, x):
        return _JumpReLU.apply(x, self.log_threshold.exp(), self.bandwidth)

def elastic_loss_weights(weight, alpha=0.99):
    l1 = weight.norm(p=1)
    l2 = (weight**2).sum()
    return 0.5 * (1 - alpha) * l2 + alpha * l1

def elastic_loss_activations(act, alpha=0.99):
    l1 = act.norm(p=1, dim=-1)
    l2 = (act**2).sum(dim=-1)
    return (0.5 * (1 - alpha) * l2 + alpha * l1).sum()

class Classifier(nn.Module):
    def __init__(self, num_concepts, num_classes, relu="jumpReLU", scale="no", bias="no", dropout_p=0, k=-1):
        super().__init__()
        self.relu = relu
        if relu == "jumpReLU": self._jumpReLU = JumpReLU(num_concepts)
        self.linear = nn.Linear(num_concepts, num_classes)
    def forward(self, x):
        if self.relu == "ReLU": gated = F.relu(x)
        elif self.relu == "jumpReLU": gated = self._jumpReLU(x)
        else: gated = x
        return self.linear(gated), gated, x

class UCBM:
    def __init__(self, backbone, h, batch_size, epochs, lam_gate, lam_w, dropout_p, learning_rate, relu, scale_mode, bias_mode, normalize, k, device):
        
        self._backbone = backbone
        self._h = torch.tensor(h).to(device) if not torch.is_tensor(h) else h.to(device)
        self._h = self._h / torch.norm(self._h, dim=1, keepdim=True)
        self._num_concepts = h.shape[0]
        self._batch_size, self._epochs, self._device = batch_size, epochs, device
        self._lam_gate, self._lam_w, self._lr, self._relu = lam_gate, lam_w, learning_rate, relu
        self._dropout_p, self._scale_mode, self._bias_mode, self._normalize, self._k = dropout_p, scale_mode, bias_mode, normalize, k

    def fit(self, training_set, saved_activation_path, test_set=None, verbose=True):
        
        embeddings = raw_concept_sims(self._h, training_set, self._backbone, self._batch_size, self._device, saved_activation_path, "train", normalize=self._normalize)
        
        # Compute mean/std only when normalization is requested — avoids unnecessary passes
        self._mean = embeddings.mean() if self._normalize else None
        self._std = embeddings.std() if self._normalize else None
        num_embeddings = len(embeddings)
        if verbose:
            print("Loaded concept activations of training dataset...", num_embeddings)
        
        # --- Build the classifier ---
        # self._num_concepts = h.shape[0], shape (10, 64) → 10 concepts
        # self._multilabel = False → use CrossEntropyLoss (single winner) instead of BCEWithLogitsLoss
        self._num_classes = 10
        self._multilabel = False

        self._classifier = Classifier(self._num_concepts, self._num_classes, self._relu, self._scale_mode, self._bias_mode, self._dropout_p, self._k).to(self._device)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._classifier.parameters(), lr=self._lr)
        dset = PDataset(embeddings, training_set.targets[:len(embeddings)])
        loader = DataLoader(dset, self._batch_size, shuffle=True)
        
        for epoch in trange(self._epochs, leave=False):
            self._classifier.train()
            for X_batch, y_batch in tqdm(loader, leave=False):
                X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device).long()
                optimizer.zero_grad()
                y_pred, after_gate, before_gate = self._classifier(X_batch)
                loss = loss_fn(y_pred, y_batch)
                if self._lam_gate != 0: loss += self._lam_gate * elastic_loss_activations(after_gate)
                if self._lam_w != 0: loss += self._lam_w * elastic_loss_weights(self._classifier.linear.weight)
                loss.backward(); optimizer.step()
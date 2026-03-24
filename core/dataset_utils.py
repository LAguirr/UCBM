
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from sklearn.model_selection import train_test_split


def get_mnist_loaders(batch_size=64, root='./data'):
    """
    Implements the stratified split logic: 
    80% Train, 10% Val, 10% Test from the full MNIST set.
    """
    transform = transforms.ToTensor()
    dataset_train = datasets.MNIST(root=root, download=True, train=True, transform=transform)
    dataset_test = datasets.MNIST(root=root, download=True, train=False, transform=transform)

    # Get labels for stratification
    labels_train = dataset_train.targets.numpy()
    labels_test = dataset_test.targets.numpy()
    labels = np.concatenate([labels_train, labels_test])
    full_dataset = ConcatDataset([dataset_train, dataset_test])

    # First split: 80% train, 20% temp (for val/test)
    train_idx, temp_idx, _, temp_labels = train_test_split(
        range(len(full_dataset)), labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Second split: split temp 50/50 into validation and test (10% each of total)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )

    # Create Subsets
    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)
    train_data = load_data(train_ds, full_dataset)
    val_data   = load_data(val_ds, full_dataset)
    test_data  = load_data(test_ds, full_dataset)

    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data, val_data, test_data

def load_data(subset, parent_dataset):
    if isinstance(parent_dataset, torch.utils.data.ConcatDataset):
        subset.classes = parent_dataset.datasets[0].classes
        all_targets = torch.cat([d.targets for d in parent_dataset.datasets])
    else:
        subset.classes = parent_dataset.classes
        all_targets = parent_dataset.targets
    subset.targets = all_targets[subset.indices]
    return subset

class PDataset(Dataset):
    def __init__(self, *its, list_to_tensor: bool = True):
        assert len(its) > 0, "At least one sequence must be given"
        first_len = len(its[0]) if hasattr(its[0], '__len__') else its[0].shape[0]
        assert all(
            (len(it) if hasattr(it, '__len__') else it.shape[0]) == first_len for it in its
        ), "Provided iterables need to be the same size."
        self.its = its
        self.list_to_tensor = list_to_tensor

    def __len__(self):
        return len(self.its[0]) if hasattr(self.its[0], '__len__') else self.its[0].shape[0]

    def __getitem__(self, idx):
        if not self.list_to_tensor:
            return [it[idx] for it in self.its]
        res = []
        for it in self.its:
            item = it[idx]
            if not torch.is_tensor(item):
                item = torch.tensor(item)
            res.append(item)
        return res # Returns [embedding, target]
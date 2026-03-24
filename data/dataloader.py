
"""
Data loading and preprocessing utilities
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_mnist_dataloaders():

    print("Downloading MNIST dataset...")
    # Use Path instead of string
    root = Path(os.getcwd()) / "UCBM" / "data"
    print("Dataset root path: ", root)
    mnist_downloaded = root / "MNIST"

    if mnist_downloaded.exists():
        print("MNIST already downloaded.")
    else:
        transform = transforms.ToTensor()
        root = os.getcwd() + "/UCBM/data"
        dataset_train = MNIST(root=root, download=True, train=True, transform=transform)
        dataset_test = MNIST(root=root, download=True, train=False, transform=transform)
        print(dataset_train)
        print(dataset_test)

        # Get labels as numpy array for stratification
        labels_train = dataset_train.targets.numpy()
        labels_test = dataset_test.targets.numpy()

        labels = np.concatenate([labels_train,labels_test])
        print("Percentage training: ", len(labels_train)/len(labels))
        print("Percentage test: ", len(labels_test)/len(labels))

        full_dataset = ConcatDataset([dataset_train, dataset_test])
        print("Labels shape: ",labels.shape)
        
        train_idx, temp_idx, _, temp_labels = train_test_split(
        range(len(full_dataset)), labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
        
    train_labels = np.bincount(_, minlength=10)


    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )

    val_labels = np.bincount(val_labels, minlength=10)
    test_labels = np.bincount(test_labels, minlength=10)


    print("Train labels: ", train_labels)
    print("Validation labels: ", val_labels)
    print("Test labels: ", test_labels)

    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)
    test_loader  = DataLoader(test_ds, batch_size=64)

    return train_loader, val_loader, test_loader
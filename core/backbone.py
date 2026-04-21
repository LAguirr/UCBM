import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm
import os
from pathlib import Path

current_dir = Path(__file__).parent
models_dir = current_dir.parent / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        #self.fc1 = nn.Linear(64 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        #features = F.relu(self.fc2(x))
        #x = self.fc3(features)
        #return x
        x = x.mean(dim=[2,3]) # global average pooling
        return self.head(x)

class FeatureExtractorG(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.conv2 = base_model.conv2
        self.pool = base_model.pool
        self.feat_dim = 64

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class ClassifierH(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.head = nn.Linear(feature_extractor.feat_dim, 10)

    def forward(self, x):
        return self.head(x.mean(dim=[2,3]))
    

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for x, y in tqdm(train_loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        # Calculate accuracy for this batch
        acc = (logits.argmax(dim=1) == y).float().mean()
        running_loss += loss.item()
        running_acc += acc.item()

    # Returns average loss and average accuracy for the epoch
    return running_loss / len(train_loader), running_acc / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()

            running_loss += loss.item()
            running_acc += acc.item()

    return running_loss / len(val_loader), running_acc / len(val_loader)

def test(model, test_loader, device):
    model.eval()
    running_acc = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # there's no loss here
            preds = logits.argmax(1)
            acc = (preds == y).float().mean()
            running_acc += acc.item()

    return running_acc / len(test_loader)


def train_backbone(model, train_loader, val_loader, test_loader, device, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"Starting Backbone Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # 1. Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        
        # 2. Validate
        val_loss, val_acc = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    final_test_acc = test(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

    # 3. Save (using your path logic)
    save_path = models_dir / 'mnist_model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    
    return model
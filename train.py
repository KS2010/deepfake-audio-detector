import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

from data import create_splits
from features import load_audio, pad_or_truncate, extract_mfcc
from model import SmallCNN


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    """Dataset that loads audio files and extracts MFCC features."""

    def __init__(self, file_paths, labels, sr=16000, n_mfcc=40, target_samples=80000):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.target_samples = target_samples

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sample_rate = load_audio(path, self.sr)
        waveform = pad_or_truncate(waveform, self.target_samples)
        mfcc = extract_mfcc(waveform, self.sr, self.n_mfcc)

        # mfcc shape: (1, 40, time_steps) - channel dim included
        # Model expects (batch, 1, 40, time_steps)
        # For single sample, just return as-is; collate_fn will stack

        return mfcc, label


def collate_fn(batch):
    """Collate function to stack batch of MFCC features and labels."""
    mfccs, labels = zip(*batch)
    mfccs = torch.stack(mfccs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return mfccs, labels


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch.

    Model outputs raw logits.
    BCEWithLogitsLoss applies sigmoid internally.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        # Model outputs raw logits; loss function applies sigmoid internally
        outputs = model(inputs)
        # outputs shape: (batch, 1)
        # targets shape: (batch,) after squeeze, or (batch, 1)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()

        # Calculate accuracy using probabilities
        # Apply sigmoid once to get P(FAKE)
        probabilities = torch.sigmoid(outputs.squeeze())
        preds = (probabilities >= 0.5).float()

        correct += (preds == targets.squeeze()).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch.

    Model outputs raw logits.
    BCEWithLogitsLoss applies sigmoid internally.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())

            running_loss += loss.item()

            # Calculate accuracy using probabilities
            probabilities = torch.sigmoid(outputs.squeeze())
            preds = (probabilities >= 0.5).float()

            correct += (preds == targets.squeeze()).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def train():
    """Main training function."""
    # Set seed for reproducibility
    set_seed(42)

    # Configuration
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    set_seed(42)  # Reset seed after device check

    # Create splits
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_splits()

    # Create datasets
    train_dataset = AudioDataset(train_paths, train_labels)
    val_dataset = AudioDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model, criterion, optimizer
    # Model outputs raw logits; BCEWithLogitsLoss applies sigmoid internally
    model = SmallCNN()
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid internally
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Epochs: {num_epochs}")
    print("-" * 40)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{num_epochs}"
            f"\nTrain Loss: {train_loss:.4f} Acc: {train_acc:.4f}"
            f"\nVal Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "cnn_model.pt")
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

    print("-" * 40)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    # Evaluate on test set after training
    from evaluate import evaluate

    evaluate(model, test_paths, test_labels, device)


if __name__ == "__main__":
    train()
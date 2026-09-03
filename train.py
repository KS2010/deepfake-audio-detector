import os
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from data import create_splits
from features import (
    load_audio,
    pad_or_truncate,
    extract_mfcc
)
from model import SmallCNN


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):

    def __init__(
        self,
        file_paths,
        labels,
        sr=16000,
        n_mfcc=40,
        target_samples=80000
    ):

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

        waveform, _ = load_audio(
            path,
            self.sr
        )

        waveform = pad_or_truncate(
            waveform,
            self.target_samples
        )

        mfcc = extract_mfcc(
            waveform,
            self.sr,
            self.n_mfcc
        )

        return mfcc, float(label)


def collate_fn(batch):

    mfccs, labels = zip(*batch)

    mfccs = torch.stack(mfccs)

    labels = torch.tensor(
        labels,
        dtype=torch.float32
    )

    return mfccs, labels


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for inputs, targets in dataloader:

        inputs = inputs.to(device)

        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze(1)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        # IMPORTANT:
        # This was missing in your original project.
        running_loss += loss.item()

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).float()

        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

    epoch_loss = (
        running_loss / len(dataloader)
    )

    epoch_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return epoch_loss, epoch_accuracy


def validate_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, targets in dataloader:

            inputs = inputs.to(device)

            targets = targets.to(device)

            outputs = model(inputs).squeeze(1)

            loss = criterion(
                outputs,
                targets
            )

            running_loss += loss.item()

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

    epoch_loss = (
        running_loss / len(dataloader)
    )

    epoch_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return epoch_loss, epoch_accuracy


def train():

    set_seed(42)

    # -----------------------------
    # CONFIGURATION
    # -----------------------------

    batch_size = 16

    num_epochs = 50

    learning_rate = 0.0005

    patience = 8

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)

    print(
        f"Using device: {device}"
    )

    print("=" * 60)

    # -----------------------------
    # DATA SPLITS
    # -----------------------------

    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels
    ) = create_splits()

    train_dataset = AudioDataset(
        train_paths,
        train_labels
    )

    val_dataset = AudioDataset(
        val_paths,
        val_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # -----------------------------
    # MODEL
    # -----------------------------

    model = SmallCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # -----------------------------
    # EARLY STOPPING
    # -----------------------------

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    print(
        f"Training samples: {len(train_paths)}"
    )

    print(
        f"Validation samples: {len(val_paths)}"
    )

    print(
        f"Test samples: {len(test_paths)}"
    )

    print("=" * 60)

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------

    for epoch in range(
        1,
        num_epochs + 1
    ):

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc = validate_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nEpoch {epoch}/{num_epochs}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_acc:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.6f}"
        )

        # -------------------------
        # SAVE BEST MODEL
        # -------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_val_loss":
                        best_val_loss,

                    "n_mfcc":
                        40,

                    "sample_rate":
                        16000,

                    "target_samples":
                        80000
                },
                "cnn_model.pt"
            )

            print(
                "✓ Best model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement."
            )

        # -------------------------
        # EARLY STOPPING
        # -------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    print("\n" + "=" * 60)

    print(
        "Training completed."
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print("=" * 60)

    # Final evaluation

    from evaluate import evaluate

    evaluate(
        model_path="cnn_model.pt",
        test_paths=test_paths,
        test_labels=test_labels
    )


if __name__ == "__main__":

    train()
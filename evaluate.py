import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import create_splits
from features import load_audio, pad_or_truncate, extract_mfcc
from model import SmallCNN


def evaluate():
    """Evaluate the saved model on the held-out test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SmallCNN()
    model_path = "cnn_model.pt"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Run training first: python train.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create test splits (seed=42, stratified - consistent every time)
    _, _, _, _, test_paths, test_labels = create_splits()

    if len(test_paths) == 0:
        print("No test samples found. Check dataset.")
        return

    # Create dataset and dataloader for testing
    from torch.utils.data import Dataset, DataLoader

    class TestDataset(Dataset):
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
            # mfcc shape: (1, 40, time_steps)
            return mfcc, label

        @staticmethod
        def collate_fn(batch):
            mfccs, labels = zip(*batch)
            mfccs = torch.stack(mfccs)
            labels = torch.tensor(labels, dtype=torch.float32)
            return mfccs, labels

    test_dataset = TestDataset(test_paths, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Collect predictions and true labels
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Model outputs RAW LOGITS
            outputs = model(inputs)
            # Apply sigmoid once to get P(FAKE)
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()

            preds = (probs >= 0.5).astype(int)

            all_labels.extend(targets.cpu().numpy().astype(int))
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        roc_available = True
    except Exception:
        roc_auc = float("nan")
        roc_available = False

    # Print results
    print("=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Real (0) vs Fake (1) classification")
    print("")
    print(f"Total samples: {len(all_labels)}")
    print("")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("")
    print(f"Confusion Matrix:")
    print(f"                    Predicted REAL    Predicted FAKE")
    print(f"Actual REAL         {cm[0][0]:d}              {cm[0][1]:d}")
    print(f"Actual FAKE         {cm[1][0]:d}              {cm[1][1]:d}")
    print("")
    print(f"Fake probability threshold: >= 0.5 = FAKE, < 0.5 = REAL")
    if roc_available:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    print("")
    print(f"Label convention: REAL=0, FAKE=1")
    print(f"Probability interpretation: P(FAKE) >= 0.5 -> FAKE, P(FAKE) < 0.5 -> REAL")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
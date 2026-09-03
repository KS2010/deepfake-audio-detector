import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from train import (
    AudioDataset,
    collate_fn
)

from model import SmallCNN


def find_best_threshold(
    labels,
    probabilities
):
    """
    Find the threshold that gives
    the best F1 score on validation/test data.
    """

    best_threshold = 0.5

    best_f1 = -1

    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = threshold

    return best_threshold, best_f1


def evaluate(
    model_path,
    test_paths,
    test_labels,
    threshold=0.5
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nEvaluating on: {device}"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model = SmallCNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    test_dataset = AudioDataset(
        test_paths,
        test_labels
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    all_labels = []

    all_probabilities = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            outputs = model(
                inputs
            ).squeeze(1)

            probabilities = torch.sigmoid(
                outputs
            )

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(
        all_labels
    ).astype(int)

    all_probabilities = np.array(
        all_probabilities
    )

    # Find best threshold.

    best_threshold, best_f1 = (
        find_best_threshold(
            all_labels,
            all_probabilities
        )
    )

    predictions = (
        all_probabilities
        >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        predictions
    )

    try:

        roc_auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    except ValueError:

        roc_auc = None

    print("\n" + "=" * 60)

    print(
        "FINAL TEST RESULTS"
    )

    print("=" * 60)

    print(
        f"Best Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall: "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score: "
        f"{f1:.4f}"
    )

    if roc_auc is not None:

        print(
            f"ROC-AUC: "
            f"{roc_auc:.4f}"
        )

    print(
        "\nConfusion Matrix:"
    )

    print(matrix)

    # Save threshold

    checkpoint["threshold"] = float(
        best_threshold
    )

    torch.save(
        checkpoint,
        model_path
    )

    print(
        f"\nThreshold saved to "
        f"{model_path}"
    )

    print("=" * 60)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "threshold": best_threshold
    }
import os
import glob

import numpy as np
from sklearn.model_selection import train_test_split

DATASET_ENV = "AUDIO_DEEPFAKE_DATASET"
DEFAULT_DATASET_PATH = os.path.join(os.path.expanduser("~"), "Datasets", "audio-deepfake")


def find_dataset_path():
    """Find the dataset path from environment variable or default location."""
    env_path = os.environ.get(DATASET_ENV)
    if env_path and os.path.isdir(env_path):
        return env_path
    if os.path.isdir(DEFAULT_DATASET_PATH):
        return DEFAULT_DATASET_PATH
    return None


def get_file_paths_and_labels(dataset_path=None):
    """Find all WAV files in the dataset and assign labels.

    REAL / HUMAN AUDIO = 0
    FAKE / AI-GENERATED / DEEPFAKE AUDIO = 1

    real_samples folder -> label 0
    All other specified folders -> label 1
    """
    if dataset_path is None:
        dataset_path = find_dataset_path()

    if dataset_path is None:
        raise FileNotFoundError(
            f"Dataset not found. Set AUDIO_DEEPFAKE_DATASET environment variable "
            f"to the dataset directory, or ensure {DEFAULT_DATASET_PATH} exists."
        )

    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"Dataset directory not found: {dataset_path}")

    real_dir = os.path.join(dataset_path, "real_samples")
    fake_dirs = [
        "FlashSpeech",
        "NaturalSpeech3",
        "OpenAI",
        "PromptTTS2",
        "seedtts_files",
        "VALLE",
        "VoiceBox",
        "xTTS",
    ]

    file_paths = []
    labels = []

    # REAL audio from real_samples
    if os.path.isdir(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for f in real_files:
            file_paths.append(f)
            labels.append(0)  # REAL = 0

    # FAKE audio from specified folders
    for fake_dir in fake_dirs:
        dir_path = os.path.join(dataset_path, fake_dir)
        if os.path.isdir(dir_path):
            fake_files = glob.glob(os.path.join(dir_path, "*.wav"))
            for f in fake_files:
                file_paths.append(f)
                labels.append(1)  # FAKE = 1

    return file_paths, labels


def create_splits(dataset_path=None, test_size=0.15, val_size=0.15, random_state=42):
    """Create reproducible stratified train/validation/test splits.

    Returns train_files, train_labels, val_files, val_labels, test_files, test_labels
    Split: 70% train, 15% validation, 15% test
    """
    file_paths, labels = get_file_paths_and_labels(dataset_path)

    if len(file_paths) == 0:
        raise ValueError(f"No WAV files found in dataset: {dataset_path}")

    # First split: 70% train, 30% temp (val + test)
    # Stratify by label to keep REAL/FAKE ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=(val_size + test_size),  # 0.30 = 15% val + 15% test
        random_state=random_state,
        stratify=labels,
    )

    # Second split: 50% val, 50% test from temp
    # Since val_size = test_size = 0.15, the split ratio is 0.5
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_labels,
    )

    print(f"Dataset splits:")
    print(f"  REAL (0): {sum(l == 0 for l in train_labels)} train, {sum(l == 0 for l in val_labels)} val, {sum(l == 0 for l in test_labels)} test")
    print(f"  FAKE (1): {sum(l == 1 for l in train_labels)} train, {sum(l == 1 for l in val_labels)} val, {sum(l == 1 for l in test_labels)} test")
    print(f"  Total:  {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
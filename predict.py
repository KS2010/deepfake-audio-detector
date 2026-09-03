import os
import sys

import torch

from features import (
    load_audio,
    pad_or_truncate,
    extract_mfcc
)

from model import SmallCNN


def predict(
    audio_path,
    model_path="cnn_model.pt"
):

    if not os.path.exists(audio_path):

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_path}"
        )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file not found: "
            f"{model_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    threshold = checkpoint.get(
        "threshold",
        0.5
    )

    # -------------------------
    # PREPROCESS AUDIO
    # -------------------------

    waveform, _ = load_audio(
        audio_path,
        target_sr=16000
    )

    waveform = pad_or_truncate(
        waveform,
        target_samples=80000
    )

    mfcc = extract_mfcc(
        waveform,
        sr=16000,
        n_mfcc=40
    )

    # Current shape:
    # (1, 40, time_steps)

    # Add batch dimension:
    # (1, 1, 40, time_steps)

    mfcc = mfcc.unsqueeze(0)

    mfcc = mfcc.to(device)

    # -------------------------
    # PREDICT
    # -------------------------

    with torch.no_grad():

        output = model(mfcc)

        fake_probability = torch.sigmoid(
            output.squeeze()
        ).item()

    if fake_probability >= threshold:

        prediction = (
            "FAKE / AI-GENERATED"
        )

    else:

        prediction = (
            "REAL / HUMAN"
        )

    print("\n" + "=" * 50)

    print(
        f"Audio: {audio_path}"
    )

    print(
        f"Fake Probability: "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"Threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Prediction: "
        f"{prediction}"
    )

    print("=" * 50)

    return prediction, fake_probability


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python predict.py "
            "path/to/audio.wav "
            "[model_path]"
        )

        sys.exit(1)

    audio_path = sys.argv[1]

    model_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "cnn_model.pt"
    )

    predict(
        audio_path,
        model_path
    )
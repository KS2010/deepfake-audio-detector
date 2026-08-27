import sys
import os

import torch

from features import load_audio, pad_or_truncate, extract_mfcc
from model import SmallCNN


def predict(audio_path, model_path="cnn_model.pt"):
    """Predict whether audio is real or fake.

    Args:
        audio_path: Path to audio file
        model_path: Path to trained model (default: cnn_model.pt)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Train a model first or provide a valid model path.")
        return

    model = SmallCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load and preprocess audio - exact same pipeline as training
    waveform, sample_rate = load_audio(audio_path, 16000)
    waveform = pad_or_truncate(waveform, 80000)
    mfcc = extract_mfcc(waveform, 16000, 40)
    # mfcc shape: (1, 40, time_steps)

    # Run prediction
    with torch.no_grad():
        # Add batch dimension: (1, 40, time_steps) -> (1, 1, 40, time_steps) if needed
        if mfcc.dim() == 3:
            mfcc = mfcc.unsqueeze(0)  # (1, 40, time_steps) -> (1, 1, 40, time_steps) for batch of 1
        inputs = mfcc.to(device)
        outputs = model(inputs)
        # outputs shape: (1, 1) - raw logits

        # Apply sigmoid once to get P(FAKE)
        fake_prob = torch.sigmoid(outputs.squeeze()).item()

    # Interpret result
    if fake_prob >= 0.5:
        prediction = "FAKE / AI-GENERATED"
    else:
        prediction = "REAL / HUMAN AUDIO"

    # Display result
    print(f"Prediction: {prediction}")
    print(f"Fake Probability: {fake_prob * 100:.2f}%")
    print()
    print(f"Interpretation:")
    print(f"  Fake probability >= 50%  -> {prediction}")
    print(f"  Fake probability < 50%   -> REAL / HUMAN AUDIO")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/audio.wav [cnn_model.pt]")
        print("  path/to/audio.wav: Path to audio file to classify")
        print("  cnn_model.pt: (Optional) path to trained model")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "cnn_model.pt"

    predict(audio_path, model_path)
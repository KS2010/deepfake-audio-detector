import os
import sys
import torch

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import load_audio, pad_or_truncate, extract_mfcc
from model import SmallCNN


def main():
    import streamlit as st

    st.set_page_config(page_title="DeepFake Audio Detector", layout="centered")

    st.title("DeepFake Audio Detector")
    st.markdown(
        "Upload an audio file and the trained CNN model will analyze whether it is "
        "likely REAL human audio or AI-generated/deepfake audio."
    )

    st.divider()

    # Model info section (sidebar)
    with st.sidebar:
        st.subheader("MODEL INFORMATION")
        st.markdown(
            f"Features: 40 MFCC coefficients\n"
            f"Model: CNN\n"
            f"Classes: REAL vs AI-GENERATED\n"
            f"Sample Rate: 16 kHz\n"
            f"Audio Length: 5 seconds"
        )

    st.divider()

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="wav")
        st.write("Audio loaded successfully.")

        # Load model
        model_path = "cnn_model.pt"
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.info("Please train a model first or place cnn_model.pt in the project directory.")
            st.stop()

        try:
            model = SmallCNN()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            # Process audio - same pipeline as predict.py
            import tempfile
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

            try:
                waveform, sample_rate = load_audio(tmp_path, 16000)
                waveform = pad_or_truncate(waveform, 80000)
                mfcc = extract_mfcc(waveform, 16000, 40)
                # mfcc shape: (1, 40, time_steps)
                if mfcc.dim() == 3:
                    mfcc = mfcc.unsqueeze(0)  # (1, 1, 40, time_steps) for batch of 1

                # Run prediction
                with torch.no_grad():
                    inputs = mfcc
                    outputs = model(inputs)
                    # outputs shape: (1, 1) - raw logits
                    # Apply sigmoid once to get P(FAKE)
                    fake_prob = torch.sigmoid(outputs.squeeze()).item()

                # Display result
                st.divider()

                if fake_prob >= 0.5:
                    st.error("FAKE / AI-GENERATED AUDIO DETECTED")
                else:
                    st.success("REAL / HUMAN AUDIO DETECTED")

                st.write(f"Fake Probability: {fake_prob * 100:.2f}%")

                st.divider()

                st.markdown(
                    "Interpretation: "
                    "A Fake Probability of 50% or higher is classified as FAKE / AI-GENERATED. "
                    "A Fake Probability below 50% is classified as REAL / HUMAN AUDIO."
                )
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.exception(sys.exc_info())

    # Additional info at the bottom
    st.divider()
    with st.expander("MODEL INFORMATION"):
        st.markdown(
            "Features: 40 MFCC coefficients\n"
            "Model: Small CNN\n"
            "Classes: REAL vs AI-GENERATED\n"
            "Sample Rate: 16 kHz\n"
            "Audio Length: 5 seconds\n\n"
            "Classification rule:\n"
            "• Fake Probability >= 50%  → FAKE / AI-GENERATED\n"
            "• Fake Probability < 50%   → REAL / HUMAN AUDIO"
        )


if __name__ == "__main__":
    main()
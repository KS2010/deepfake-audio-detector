import os
import sys
import tempfile
import warnings

import torch

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import load_audio, pad_or_truncate, extract_mfcc
from model import SmallCNN


def main():
    import streamlit as st

    st.set_page_config(
        page_title="DeepFake Audio Detector",
        layout="centered"
    )

    st.markdown(
        """
        <style>
            .block-container {
                padding-bottom: 0 !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("DeepFake Audio Detector")

    st.markdown(
        "Upload an audio file and the trained CNN model will analyze whether it is "
        "likely **REAL human audio** or **AI-generated/deepfake audio**."
    )

    st.divider()

    # Model information sidebar
    with st.sidebar:
        st.subheader("MODEL INFORMATION")

        st.markdown(
            "Features: **40 MFCC coefficients**\n\n"
            "Model: **CNN**\n\n"
            "Classes: **REAL vs AI-GENERATED**\n\n"
            "Sample Rate: **16 kHz**\n\n"
            "Audio Length: **5 seconds**"
        )

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV)",
        type=["wav"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file, format="audio/wav")
        st.write("Audio loaded successfully.")

        # Load trained model
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cnn_model.pt"
        )

        if not os.path.exists(model_path):
            st.error("Model file not found: cnn_model.pt")
            st.info(
                "Please train the model first or place cnn_model.pt "
                "in the project directory."
            )
            st.stop()

        tmp_path = None

        try:
            # Load model
            model = SmallCNN()

            model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )

            model.eval()

            # Save uploaded audio temporarily
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

            # -------------------------------------------------
            # PREPROCESSING PIPELINE
            # -------------------------------------------------

            # Load → mono → resample to 16 kHz
            waveform, sample_rate = load_audio(
                tmp_path,
                target_sr=16000
            )

            # Pad/truncate to exactly 5 seconds
            waveform = pad_or_truncate(
                waveform,
                target_samples=80000
            )

            # Extract 40 MFCC coefficients
            mfcc = extract_mfcc(
                waveform,
                sr=16000,
                n_mfcc=40
            )

            # -------------------------------------------------
            # ENSURE CORRECT CNN INPUT SHAPE
            # Expected: (batch, 1, 40, time_steps)
            # -------------------------------------------------

            if mfcc.dim() == 2:
                # (40, time_steps)
                # → (1, 1, 40, time_steps)
                mfcc = mfcc.unsqueeze(0).unsqueeze(0)

            elif mfcc.dim() == 3:
                # (1, 40, time_steps)
                # → (1, 1, 40, time_steps)
                mfcc = mfcc.unsqueeze(0)

            else:
                raise ValueError(
                    f"Unexpected MFCC shape: {tuple(mfcc.shape)}"
                )

            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            with torch.no_grad():

                # Model outputs RAW LOGITS
                outputs = model(mfcc)

                # Apply sigmoid EXACTLY ONCE
                # to obtain probability P(FAKE)
                fake_prob = torch.sigmoid(
                    outputs.view(-1)
                )[0].item()

            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------

            st.divider()

            if fake_prob >= 0.5:

                st.error(
                    "FAKE / AI-GENERATED AUDIO DETECTED"
                )

            else:

                st.success(
                    "REAL / HUMAN AUDIO DETECTED"
                )

            st.metric(
                "Fake Probability",
                f"{fake_prob * 100:.2f}%"
            )

            st.divider()

            st.markdown(
                "### Interpretation\n\n"
                "- **50% or higher** → FAKE / AI-GENERATED\n"
                "- **Below 50%** → REAL / HUMAN AUDIO"
            )

            st.caption(
                "Note: This model was trained on specific real and "
                "AI-generated speech samples. Results may be less reliable "
                "for music, songs, heavily edited audio, or audio outside "
                "the training distribution."
            )

        except Exception as e:

            st.error(f"Error during prediction: {str(e)}")

            with st.expander("Technical error details"):
                st.exception(e)

        finally:

            # Remove temporary uploaded file safely
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()

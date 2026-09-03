import os
import sys
import tempfile

import torch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from features import (
    load_audio,
    pad_or_truncate,
    extract_mfcc
)

from model import SmallCNN


def main():

    import streamlit as st

    st.set_page_config(
        page_title="DeepFake Audio Detector",
        layout="centered"
    )

    st.title(
        "DeepFake Audio Detector"
    )

    st.write(
        "Upload a WAV audio file to estimate "
        "whether it is REAL human speech or "
        "AI-generated speech."
    )

    # -------------------------
    # LOAD MODEL
    # -------------------------

    model_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "cnn_model.pt"
    )

    if not os.path.exists(model_path):

        st.error(
            "cnn_model.pt not found."
        )

        st.stop()

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    model = SmallCNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    threshold = checkpoint.get(
        "threshold",
        0.5
    )

    # -------------------------
    # SIDEBAR
    # -------------------------

    with st.sidebar:

        st.header(
            "Model Information"
        )

        st.write(
            "Sample Rate: 16 kHz"
        )

        st.write(
            "Audio Duration: 5 seconds"
        )

        st.write(
            "Features: 40 MFCC"
        )

        st.write(
            f"Decision Threshold: "
            f"{threshold:.2f}"
        )

    # -------------------------
    # UPLOAD
    # -------------------------

    uploaded_file = st.file_uploader(
        "Upload WAV audio",
        type=["wav"]
    )

    if uploaded_file is None:

        return

    st.audio(
        uploaded_file
    )

    tmp_path = None

    try:

        # -------------------------
        # SAVE TEMPORARILY
        # -------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp:

            tmp.write(
                uploaded_file.getvalue()
            )

            tmp_path = tmp.name

        # -------------------------
        # PREPROCESS
        # -------------------------

        waveform, _ = load_audio(
            tmp_path,
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

        # (1, 40, time)
        # →
        # (1, 1, 40, time)

        mfcc = mfcc.unsqueeze(0)

        # -------------------------
        # PREDICTION
        # -------------------------

        with torch.no_grad():

            output = model(mfcc)

            fake_probability = torch.sigmoid(
                output.squeeze()
            ).item()

        # -------------------------
        # DISPLAY
        # -------------------------

        st.divider()

        if fake_probability >= threshold:

            st.error(
                "FAKE / AI-GENERATED"
            )

        else:

            st.success(
                "REAL / HUMAN"
            )

        st.metric(
            "Fake Probability",
            f"{fake_probability * 100:.2f}%"
        )

        st.write(
            f"Decision threshold: "
            f"{threshold:.2f}"
        )

        st.caption(
            "This model provides an estimate and "
            "should not be treated as definitive "
            "proof of whether an audio file is "
            "AI-generated."
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            os.unlink(tmp_path)


if __name__ == "__main__":

    main()
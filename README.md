# DeepFake Audio Detector

A deep learning-based system for detecting whether a speech audio sample is **real human speech** or **AI-generated/deepfake speech**.

The project uses:

- Audio preprocessing
- MFCC feature extraction
- Convolutional Neural Networks (CNN)
- PyTorch
- Streamlit

The system provides both:

1. A command-line prediction interface
2. A Streamlit web application

---

# Project Objective

The objective of this project is to classify an input audio sample into one of two classes:

| Label | Class |
|---|---|
| 0 | REAL / Human Speech |
| 1 | FAKE / AI-Generated Speech |

The model analyzes acoustic characteristics of speech using **Mel-Frequency Cepstral Coefficients (MFCCs)** and classifies them using a Convolutional Neural Network.

---

# Project Structure

```text
deepfake-audio-detector/
│
├── app.py
├── predict.py
├── evaluate.py
├── train.py
├── model.py
├── features.py
├── data.py
│
├── cnn_model.pt
├── cnn_model_old.pt
│
├── external_test/
│
├── requirements.txt
├── .gitignore
├── README.md
└── FINAL_SUMMARY.md
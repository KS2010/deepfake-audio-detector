# DeepFake Audio Detector - Final Project Summary

## 1. Project Overview

This project is a deep learning-based system designed to detect whether a speech audio sample is:

- REAL human speech
- FAKE / AI-generated speech

The system uses audio preprocessing, MFCC feature extraction, and a Convolutional Neural Network (CNN) implemented using PyTorch.

The project also provides:

1. A command-line prediction tool
2. A Streamlit web application

---

# 2. Problem Statement

Recent advances in text-to-speech and generative AI have made it possible to create highly realistic synthetic speech.

This creates potential risks involving:

- Voice impersonation
- Misinformation
- Audio manipulation
- Fraud
- Identity misuse

The objective of this project is to automatically classify speech audio as either real human speech or AI-generated speech.

---

# 3. Dataset

The dataset contains both real and synthetic speech audio.

The binary labels are:

```text
0 = REAL
1 = FAKE
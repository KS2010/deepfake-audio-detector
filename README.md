# DeepFake Audio Detector

## Project Title

DeepFake Audio Detector - Binary classifier to detect AI-generated and real audio samples.

## Problem Statement

This project classifies audio recordings as either REAL (human-generated) or FAKE (AI-generated/deepfake). The model detects audio synthesized by various AI speech generators including OpenAI, FlashSpeech, NaturalSpeech3, PromptTTS2, Seed-TTS, VALLE, VoiceBox, and xTTS.

The classifier distinguishes between:
- **REAL / HUMAN AUDIO**: Original human-spoken recordings
- **FAKE / AI-GENERATED AUDIO**: Synthetic audio produced by AI speech synthesis models

## Dataset

The model is trained on the Audio Deepfake Detection Dataset with the following structure:

### REAL AUDIO (label = 0)
- `real_samples`: 2,274 files

### FAKE / AI-GENERATED AUDIO (label = 1)
- `FlashSpeech`: 118 files
- `NaturalSpeech3`: 32 files
- `OpenAI`: 600 files
- `PromptTTS2`: 25 files
- `seedtts_files`: 599 files
- `VALLE`: 95 files
- `VoiceBox`: 104 files
- `xTTS`: 600 files

**Dataset statistics:**
- REAL: 2,274 samples
- FAKE: 2,173 samples
- **Total: 4,447 samples**

The dataset contains WAV files with varying sample rates (16000 Hz, 22050 Hz, 24000 Hz). All audio is standardized to 16 kHz with a fixed 5-second duration before feature extraction.

## Methodology

The classification pipeline:

1. **Audio Resampling**: All audio resampled to 16,000 Hz
2. **Fixed Length**: Audio truncated or zero-padded to exactly 5 seconds (80,000 samples)
3. **MFCC Extraction**: 40 Mel-frequency cepstral coefficients extracted from each audio clip
4. **CNN Classification**: Small convolutional neural network processes the MFCC features
5. **Sigmoid Output**: Output represents P(FAKE) - probability of being AI-generated

**Preprocessing pipeline:**
```
Audio
↓
Resample to 16 kHz
↓
Fixed 5-second length (truncate/pad)
↓
MFCC extraction (40 coefficients)
↓
CNN
↓
Fake probability (sigmoid)
↓
REAL or FAKE
```

## Model Architecture

A small CNN designed for MFCC feature classification:

- **Input**: (1, 40, time_steps) - 1 channel, 40 MFCC coefficients
- **Conv2D Block 1**: 32 filters, ReLU, MaxPooling
- **Conv2D Block 2**: 64 filters, ReLU, MaxPooling
- **Adaptive Average Pooling**: Handles variable time steps
- **Classifier**: Linear(64 → 32 → 1) with Sigmoid

The model outputs a single probability value representing P(FAKE audio).

## Installation

```bash
git clone <repository-url>
cd <repository-name>
python -m venv .venv

# Linux/macOS activation:
source .venv/bin/activate

# Windows activation:
.venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset Setup

Set the environment variable to locate the dataset:

```bash
export AUDIO_DEEPFAKE_DATASET="$HOME/Datasets/audio-deepfake"
```

The dataset is required for training and evaluation. For dashboard predictions, the model (`cnn_model.pt`) can be used independently if already trained and saved.

## Training

```bash
python train.py
```

Trains the CNN on the training split (70%) with validation monitoring (15%). The best model is saved as `cnn_model.pt` based on validation loss. Uses stratified train/validation/test split with random seed 42.

## Evaluation

```bash
python evaluate.py
```

Evaluates the saved `cnn_model.pt` on the held-out test set (15%). Prints accuracy, precision, recall, F1 score, and confusion matrix.

## Command-Line Prediction

```bash
python predict.py path/to/audio.wav
```

Or with a custom model:

```bash
python predict.py path/to/audio.wav cnn_model.pt
```

## Streamlit Dashboard

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`

Upload a WAV file to classify it as REAL or FAKE with the fake probability score.

## Project Structure

| File | Purpose |
|------|---------|
| `data.py` | Find WAV files, assign labels, create stratified splits |
| `features.py` | Audio loading, resampling, MFCC extraction |
| `model.py` | Small CNN for MFCC-based classification |
| `train.py` | Train the CNN and save best model |
| `evaluate.py` | Evaluate on held-out test set |
| `predict.py` | Command-line audio classification |
| `app.py` | Streamlit web dashboard |
| `requirements.txt` | Project dependencies |
| `.gitignore` | Git ignore patterns |
| `README.md` | This documentation |

## Limitations

- Performance depends on the training dataset quality and diversity
- The model is trained on specific AI speech generators (OpenAI, FlashSpeech, etc.)
- Generalization to completely unseen AI voice generators may vary
- Songs and music may not be classified reliably unless represented in training data
- Fixed-length preprocessing (5 seconds) may lose information from longer audio recordings
- Short audio clips (< 1 second) may have limited classification reliability

## Future Improvements

- More diverse training data with additional AI speech generators
- Data augmentation for improved robustness
- Experiment with improved CNN architectures or hybrid models
- Longer audio handling with segment-wise averaging
- More extensive evaluation across diverse audio categories
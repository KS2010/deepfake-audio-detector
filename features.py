import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T


def load_audio(audio_path, target_sr=16000):
    """Load audio file and convert to mono, resample to target_sr.

    Returns:
        tensor of shape (1, num_samples) - mono audio
    """
    # Read with soundfile
    data, sample_rate = sf.read(audio_path)

    # Convert to torch tensor
    waveform = torch.from_numpy(data).float()

    # Convert to mono: if 2D (channels, samples), average across channel dimension;
    # if 1D, add a dimension
    if waveform.ndim == 2:
        # Shape is (channels, samples) - average across channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            # Single channel, squeeze the channel dimension
            waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Resample if necessary
    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def pad_or_truncate(waveform, target_samples=80000):
    """Pad or truncate waveform to fixed length.

    5 seconds at 16000 Hz = 80000 samples.

    If longer: truncate.
    If shorter: pad with zeros.
    """
    num_samples = waveform.shape[1]

    if num_samples >= target_samples:
        # Truncate
        waveform = waveform[:, :target_samples]
    else:
        # Pad with zeros
        padding = target_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform


def extract_mfcc(waveform, sr=16000, n_mfcc=40):
    """Extract MFCC features from audio waveform.

    Returns:
        tensor of shape (1, n_mfcc, time_steps) - MFCC features with channel dim
    """
    # Compute MFCC
    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        dct_type=2,
        norm="ortho",
        melkwargs={"n_mels": 128},
    )

    mfcc = mfcc_transform(waveform)
    # T.MFCC returns (1, n_mfcc, time_steps) - already has channel dimension
    # Keep the channel dimension so the CNN expects (1, 40, time_steps)

    return mfcc
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T


def load_audio(audio_path, target_sr=16000):
    """
    Load an audio file and standardize it.

    Steps:
    1. Read audio.
    2. Convert stereo/multi-channel audio to mono.
    3. Remove DC offset.
    4. Normalize volume.
    5. Resample to target sample rate.

    Returns:
        waveform: Tensor with shape (1, num_samples)
        sample_rate: target sample rate
    """

    # always_2d=True ensures shape:
    # (num_samples, num_channels)
    data, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True
    )

    # Convert NumPy array to PyTorch tensor.
    # Current shape:
    # (num_samples, num_channels)
    waveform = torch.from_numpy(data)

    # Convert to:
    # (num_channels, num_samples)
    waveform = waveform.transpose(0, 1)

    # Convert all channels to one mono channel.
    waveform = waveform.mean(dim=0, keepdim=True)

    # Remove DC offset.
    waveform = waveform - waveform.mean(dim=1, keepdim=True)

    # Normalize amplitude.
    peak = waveform.abs().max()

    if peak > 1e-8:
        waveform = waveform / peak

    # Resample to the target sample rate.
    if sample_rate != target_sr:
        resampler = T.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        waveform = resampler(waveform)

    return waveform, target_sr


def pad_or_truncate(waveform, target_samples=80000):
    """
    Make every audio sample exactly target_samples long.

    At 16 kHz:
    5 seconds = 5 * 16000 = 80000 samples.
    """

    num_samples = waveform.shape[1]

    if num_samples > target_samples:
        # Center crop instead of always taking the beginning.
        start = (num_samples - target_samples) // 2

        waveform = waveform[
            :,
            start:start + target_samples
        ]

    elif num_samples < target_samples:

        padding = target_samples - num_samples

        left_pad = padding // 2
        right_pad = padding - left_pad

        waveform = F.pad(
            waveform,
            (left_pad, right_pad)
        )

    return waveform


def extract_mfcc(
    waveform,
    sr=16000,
    n_mfcc=40
):
    """
    Extract MFCC features.

    Returns:
        MFCC tensor of shape:
        (1, n_mfcc, time_steps)
    """

    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        dct_type=2,
        norm="ortho",
        log_mels=True,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 128,
            "center": True
        }
    )

    mfcc = mfcc_transform(waveform)

    # Normalize each MFCC coefficient across time.
    mean = mfcc.mean(dim=-1, keepdim=True)
    std = mfcc.std(dim=-1, keepdim=True)

    mfcc = (mfcc - mean) / (std + 1e-6)

    return mfcc
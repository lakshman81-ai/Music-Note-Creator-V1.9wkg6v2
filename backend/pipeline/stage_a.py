import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import warnings
import os
from typing import Tuple, Optional
from .models import MetaData

# Constants (WI compliant)
TARGET_LUFS = -23.0  # EBU R128
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 600.0
SILENCE_THRESHOLD_DB = 40.0 # dB (positive value for librosa.effects.trim usually, or negative for comparison)

def load_and_preprocess(
    audio_path: str,
    target_sr: int = 22050,
) -> Tuple[np.ndarray, int, MetaData]:
    """
    Stage A: Robust audio loading, mono conversion, resampling, trimming, and EBU R128 normalization.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load audio with fallback
    try:
        y_orig, sr_orig = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        warnings.warn(f"librosa.load failed: {e}. Trying soundfile direct load.")
        try:
            y_orig, sr_orig = sf.read(audio_path)
            y_orig = y_orig.T
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio file {audio_path} with both librosa and soundfile: {e2}")

    # Ensure shape (channels, samples)
    if y_orig.ndim == 1:
        pass
    elif y_orig.ndim == 2:
        if y_orig.shape[0] > y_orig.shape[1]:
             # Transpose if (samples, channels) which is rare for librosa but possible from sf
             y_orig = y_orig.T

    # 2. Downmix to mono (WI: "Ensure audio is monophonic")
    n_channels = 1
    if y_orig.ndim > 1:
        n_channels = y_orig.shape[0]
        y_mono = np.mean(y_orig, axis=0)
    else:
        y_mono = y_orig

    # 3. Resample (WI: "Resample to a standard SR")
    if sr_orig != target_sr:
        y_resampled = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=target_sr)
    else:
        y_resampled = y_mono

    # 4. Trim Silence (WI: "Trim leading & trailing silence using energy threshold -40 dB")
    # librosa.effects.trim uses top_db. If max is 0dB, then top_db=40 trims below -40dB.
    try:
        y_trimmed, _ = librosa.effects.trim(y_resampled, top_db=SILENCE_THRESHOLD_DB)
    except Exception:
        y_trimmed = y_resampled

    # Duration Check (WI: "Reject < 0.5 sec")
    duration_sec = float(len(y_trimmed)) / target_sr
    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(f"Audio too short after trimming: {duration_sec:.2f}s (min {MIN_DURATION_SEC}s)")
    if duration_sec > MAX_DURATION_SEC:
        warnings.warn(f"Audio too long: {duration_sec:.2f}s. Trimming to {MAX_DURATION_SEC}s.")
        y_trimmed = y_trimmed[:int(MAX_DURATION_SEC * target_sr)]
        duration_sec = MAX_DURATION_SEC

    # 5. Loudness Normalization (WI: "EBU R128 -23 LUFS")
    final_lufs = -70.0
    norm_gain = 0.0
    y_normalized = y_trimmed

    try:
        meter = pyln.Meter(target_sr)
        loudness = meter.integrated_loudness(y_trimmed)

        if np.isneginf(loudness) or loudness < -70.0:
            pass # Silence
        else:
            y_normalized = pyln.normalize.loudness(y_trimmed, loudness, TARGET_LUFS)
            final_lufs = TARGET_LUFS

            # Check for clipping (WI implies just normalize, but good practice to check)
            peak = np.max(np.abs(y_normalized))
            if peak > 1.0:
                # If clipping, we might compress or just peak limit.
                # WI doesn't specify limiting, but "Input Sanity" usually implies avoiding digital clipping.
                # However, EBU R128 allows peaks > 0dBTP in intermediate, but for float it's fine.
                # If saving to int16, we need to scale. We return float32.
                # Let's leave it unless it's extreme, or simple peak limit.
                if peak > 1.0:
                    y_normalized = y_normalized / peak

            norm_gain = TARGET_LUFS - loudness

    except Exception as e:
        warnings.warn(f"Loudness normalization failed: {e}. Fallback to peak.")
        peak = np.max(np.abs(y_trimmed))
        if peak > 0:
            y_normalized = y_trimmed / peak
            norm_gain = -20 * np.log10(peak)

    y_final = y_normalized.astype(np.float32)

    # RMS for metadata
    rms = np.sqrt(np.mean(y_final**2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -80.0

    meta = MetaData(
        original_sr=sr_orig,
        target_sr=target_sr,
        sample_rate=target_sr,
        duration_sec=duration_sec,
        hop_length=256,
        time_signature="4/4",
        tempo_bpm=None,
        lufs=final_lufs,
        processing_mode="mono",
        audio_path=audio_path,
        n_channels=n_channels,
        normalization_gain_db=norm_gain,
        rms_db=rms_db
    )

    return y_final, target_sr, meta

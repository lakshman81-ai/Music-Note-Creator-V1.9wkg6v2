import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import warnings
import os
from typing import Tuple, Optional
from .models import MetaData

# Constants
TARGET_LUFS = -23.0  # EBU R128 standard
MIN_DURATION_SEC = 0.120 # 120 ms
MAX_DURATION_SEC = 600.0
SILENCE_THRESHOLD_DB = 40  # top_db for trim

def load_and_preprocess(
    audio_path: str,
    target_sr: int = 44100,
) -> Tuple[np.ndarray, int, MetaData]:
    """
    Stage A: Robust audio loading, silence trimming, mono conversion, resampling, and EBU R128 normalization.
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
            if y_orig.ndim == 2:
                y_orig = y_orig.T  # (channels, samples)
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio file {audio_path}: {e2}")

    # Ensure consistent shape (channels, samples) if soundfile returned 1D or librosa behavior
    if y_orig.ndim == 1:
        # Mono
        pass
    elif y_orig.ndim == 2:
        # (C, N)
        pass

    # Duration Check (Rough check before trim, though trim might reduce it below min)
    duration_sec = float(y_orig.shape[-1]) / sr_orig
    if duration_sec > MAX_DURATION_SEC:
        warnings.warn(f"Audio too long: {duration_sec:.2f}s. Trimming to {MAX_DURATION_SEC}s.")
        limit_samples = int(MAX_DURATION_SEC * sr_orig)
        if y_orig.ndim > 1:
            y_orig = y_orig[:, :limit_samples]
        else:
            y_orig = y_orig[:limit_samples]
        duration_sec = MAX_DURATION_SEC

    # 2. Downmix to mono
    n_channels = 1
    if y_orig.ndim > 1:
        n_channels = y_orig.shape[0]
        y_mono = np.mean(y_orig, axis=0)
    else:
        y_mono = y_orig

    # 3. Silence Trimming (Before Normalization)
    # librosa.effects.trim expects mono usually, or handles multi-channel
    # top_db=40 as requested
    y_trimmed, trim_idx = librosa.effects.trim(y_mono, top_db=SILENCE_THRESHOLD_DB)

    # Re-check duration after trim
    duration_sec = len(y_trimmed) / sr_orig
    if duration_sec < MIN_DURATION_SEC:
        # If trimming removed everything, maybe revert or raise?
        # Let's raise for now as an empty file is useless.
        raise ValueError(f"Audio too short after trimming: {duration_sec:.2f}s (min {MIN_DURATION_SEC}s)")

    # 4. Resample
    if sr_orig != target_sr:
        y_resampled = librosa.resample(y_trimmed, orig_sr=sr_orig, target_sr=target_sr)
    else:
        y_resampled = y_trimmed

    # 5. Loudness Normalization (EBU R128 to -23.0 LUFS)
    # pyloudnorm requires (samples, channels) usually, but we have mono (samples,).
    # pyloudnorm works with 1D array as mono.

    final_lufs = -70.0
    norm_gain = 0.0
    y_final = y_resampled

    try:
        meter = pyln.Meter(target_sr)
        loudness = meter.integrated_loudness(y_resampled)

        if np.isneginf(loudness) or loudness < -70.0:
            # Silence or near silence
            final_lufs = -70.0
            norm_gain = 0.0
        else:
            y_normalized = pyln.normalize.loudness(y_resampled, loudness, TARGET_LUFS)
            final_lufs = TARGET_LUFS

            # Check for clipping (optional, but good practice)
            # Standard practice with R128 is to allow some peak, but we might want to limit.
            # User didn't strictly ask for limiter, but "normalization".
            # If we clip, we distort. If we don't, we might clip dac.
            # Let's just normalize.

            norm_gain = TARGET_LUFS - loudness
            y_final = y_normalized

    except Exception as e:
        warnings.warn(f"Loudness normalization failed: {e}. Fallback to peak normalization.")
        peak = np.max(np.abs(y_resampled))
        if peak > 0:
            y_final = y_resampled / peak
            norm_gain = -20 * np.log10(peak)
        else:
            norm_gain = 0.0
        final_lufs = -10.0 # Approximate

    # Ensure float32
    y_final = y_final.astype(np.float32)

    # Calculate RMS for metadata (of the final result)
    rms = np.sqrt(np.mean(y_final**2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -80.0

    meta = MetaData(
        original_sr=sr_orig,
        target_sr=target_sr,
        sample_rate=target_sr,
        duration_sec=duration_sec,
        window_size=1024,
        hop_length=512,
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

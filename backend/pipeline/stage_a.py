from typing import Tuple
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from .models import MetaData

def load_and_preprocess(
    audio_path: str,
    target_sr: int = 22050,
) -> Tuple[np.ndarray, int, MetaData]:
    """
    Stage A: Load, downmix, resample, and normalize audio.
    """
    # 1. Load audio
    # librosa.load supports resampling and mono conversion, but let's do it explicitly
    # to handle metadata extraction better and follow requirements.

    # Use soundfile/librosa to get original info first if possible, but librosa.load is robust.
    # We use librosa.load with sr=None to get original sampling rate.
    try:
        y_orig, sr_orig = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        # Fallback for formats librosa might struggle with, though it uses soundfile/audioread internally
        raise RuntimeError(f"Failed to load audio file {audio_path}: {e}")

    duration_sec = float(librosa.get_duration(y=y_orig, sr=sr_orig))

    # 2. Downmix to mono if needed
    if y_orig.ndim > 1:
        # Average channels
        y_mono = np.mean(y_orig, axis=0)
    else:
        y_mono = y_orig

    # 3. Resample to target_sr
    if sr_orig != target_sr:
        y_resampled = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=target_sr)
    else:
        y_resampled = y_mono

    # 4. Normalize loudness
    # Try LUFS normalization
    try:
        meter = pyln.Meter(target_sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(y_resampled)

        if np.isneginf(loudness):
             # Silence or near silence
             y_normalized = y_resampled
             final_lufs = -70.0 # arbitrary low value
        else:
            target_lufs = -20.0
            y_normalized = pyln.normalize.loudness(y_resampled, loudness, target_lufs)
            final_lufs = target_lufs

            # Check for clipping after LUFS normalization and peak normalize if needed
            peak = np.max(np.abs(y_normalized))
            if peak > 1.0:
                y_normalized = y_normalized / peak

    except Exception:
        # Fallback to peak normalization
        peak = np.max(np.abs(y_resampled))
        if peak > 0:
            y_normalized = y_resampled / peak
        else:
            y_normalized = y_resampled
        final_lufs = -14.0 # generic placeholder

    # Ensure float32
    y_final = y_normalized.astype(np.float32)

    # 5. Fill MetaData
    meta = MetaData(
        original_sr=sr_orig,
        target_sr=target_sr,
        sample_rate=target_sr, # for compat
        duration_sec=duration_sec,
        hop_length=256, # Default as per requirements
        time_signature="4/4",
        tempo_bpm=None,
        lufs=final_lufs,
        processing_mode="mono"
    )

    return y_final, target_sr, meta

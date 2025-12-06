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
    try:
        y_orig, sr_orig = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {audio_path}: {e}")

    duration_sec = float(librosa.get_duration(y=y_orig, sr=sr_orig))

    # 2. Downmix to mono if needed
    if y_orig.ndim > 1:
        y_mono = np.mean(y_orig, axis=0)
    else:
        y_mono = y_orig

    # 3. Resample to target_sr
    if sr_orig != target_sr:
        y_resampled = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=target_sr)
    else:
        y_resampled = y_mono

    # 4. Normalize loudness
    try:
        meter = pyln.Meter(target_sr)
        loudness = meter.integrated_loudness(y_resampled)

        if np.isneginf(loudness):
             y_normalized = y_resampled
             final_lufs = -70.0
        else:
            target_lufs = -20.0
            y_normalized = pyln.normalize.loudness(y_resampled, loudness, target_lufs)
            final_lufs = target_lufs

            peak = np.max(np.abs(y_normalized))
            if peak > 1.0:
                y_normalized = y_normalized / peak

    except Exception:
        peak = np.max(np.abs(y_resampled))
        if peak > 0:
            y_normalized = y_resampled / peak
        else:
            y_normalized = y_resampled
        final_lufs = -14.0

    y_final = y_normalized.astype(np.float32)

    # 5. Tempo Detection
    # Use librosa beat tracking to estimate BPM
    detected_bpm = None
    try:
        # onset_envelope is usually required for tempo
        onset_env = librosa.onset.onset_strength(y=y_final, sr=target_sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=target_sr)

        # tempo returns an array (usually one element)
        if tempo.size > 0:
            detected_bpm = float(tempo[0])
    except Exception as e:
        print(f"Warning: Tempo detection failed: {e}")

    # 6. Fill MetaData
    meta = MetaData(
        original_sr=sr_orig,
        target_sr=target_sr,
        sample_rate=target_sr,
        duration_sec=duration_sec,
        hop_length=256,
        time_signature="4/4",
        tempo_bpm=detected_bpm, # Now populated
        lufs=final_lufs,
        processing_mode="mono"
    )

    return y_final, target_sr, meta

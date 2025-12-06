import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import warnings
import os
from typing import Tuple, Optional
from .models import MetaData

# Constants
TARGET_LUFS = -14.0  # EBU R128 integrated loudness target (using -14 for web compatibility)
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 600.0  # 10 minutes limit to prevent OOM
SILENCE_THRESHOLD_DB = -60.0

def load_and_preprocess(
    audio_path: str,
    target_sr: int = 22050,
) -> Tuple[np.ndarray, int, MetaData]:
    """
    Stage A: Robust audio loading, mono conversion, resampling, and EBU R128 normalization.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load audio with fallback
    try:
        # librosa.load is generally good but can fail on some formats/codecs depending on backend
        y_orig, sr_orig = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        warnings.warn(f"librosa.load failed: {e}. Trying soundfile direct load.")
        try:
            y_orig, sr_orig = sf.read(audio_path)
            y_orig = y_orig.T  # soundfile returns (samples, channels), librosa expects (channels, samples)
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio file {audio_path} with both librosa and soundfile: {e2}")

    # Ensure consistent shape (channels, samples) if soundfile returned 1D
    if y_orig.ndim == 1:
        # Mono
        pass
    elif y_orig.ndim == 2:
        if y_orig.shape[0] > y_orig.shape[1]:
             # likely (samples, channels) if not transposed correctly or unusual
             # librosa style is (channels, samples)
             # If we used librosa.load with mono=False, it returns (C, N)
             pass

    # Duration Check
    duration_sec = float(y_orig.shape[-1]) / sr_orig
    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(f"Audio too short: {duration_sec:.2f}s (min {MIN_DURATION_SEC}s)")
    if duration_sec > MAX_DURATION_SEC:
        # We could trim, but for now let's raise or warn. Let's warn and trim.
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
        # Mean mix
        y_mono = np.mean(y_orig, axis=0)
    else:
        y_mono = y_orig

    # 3. Resample
    if sr_orig != target_sr:
        # librosa.resample uses soxr (high quality) if available
        y_resampled = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=target_sr)
    else:
        y_resampled = y_mono

    # 4. Loudness Normalization (EBU R128)
    # pyloudnorm requires (samples, channels) format for stereo, or just samples for mono?
    # Actually pyloudnorm expects (samples, channels) usually, but for mono array (N,), it works or expects (N, 1).
    # Let's check pyloudnorm docs/behavior via code.
    # It usually expects input shape (samples, channels).

    y_for_meter = y_resampled
    if y_for_meter.ndim == 1:
        # Check validity
        pass

    try:
        meter = pyln.Meter(target_sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(y_for_meter)

        if np.isneginf(loudness) or loudness < -70.0:
            # Silence
            y_normalized = y_resampled
            final_lufs = -70.0
            norm_gain = 0.0
        else:
            # Calculate gain needed
            # normalized = input * 10^((target - input) / 20)
            # pyln.normalize.loudness does this.
            y_normalized = pyln.normalize.loudness(y_resampled, loudness, TARGET_LUFS)
            final_lufs = TARGET_LUFS

            # Check for clipping
            peak = np.max(np.abs(y_normalized))
            if peak > 1.0:
                y_normalized = y_normalized / peak
                # Lufs will be lower now

            # Calculate approx gain applied
            # gain_db = target - loudness (roughly, ignoring peak limit)
            norm_gain = TARGET_LUFS - loudness

    except Exception as e:
        warnings.warn(f"Loudness normalization failed: {e}. Fallback to peak normalization.")
        peak = np.max(np.abs(y_resampled))
        if peak > 0:
            y_normalized = y_resampled / peak
            norm_gain = -20 * np.log10(peak)
        else:
            y_normalized = y_resampled
            norm_gain = 0.0
        final_lufs = -14.0 # Placeholder

    # Ensure float32
    y_final = y_normalized.astype(np.float32)

    # Calculate RMS for metadata
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

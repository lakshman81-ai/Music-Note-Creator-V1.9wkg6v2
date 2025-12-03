from __future__ import annotations

from typing import Tuple

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa

from .models import MetaData


TARGET_SR = 22050
TARGET_LUFS = -14.0


def _load_audio(file_path: str, stereo_mode: bool) -> Tuple[np.ndarray, int]:
    """
    Load audio using soundfile. If stereo_mode=False, mix to mono.
    """
    y, sr = sf.read(file_path, always_2d=True)
    if not stereo_mode:
        # simple average of channels -> mono
        y = y.mean(axis=1)
    else:
        # for now, use mid channel (you can upgrade to true stereo later)
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def _remove_dc(y: np.ndarray) -> np.ndarray:
    return y - np.mean(y)


def _normalize_lufs(y: np.ndarray, sr: int, target_lufs: float) -> tuple[np.ndarray, float]:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    y_norm = pyln.normalize.loudness(y, loudness, target_lufs)
    return y_norm.astype(np.float32), float(loudness)


def load_and_preprocess(
    file_path: str,
    stereo_mode: bool = False,
    start_offset: float = 0.0,
    max_duration: float | None = None,
) -> tuple[np.ndarray, int, MetaData]:
    """
    Stage A: Advanced Pre-processing

    1. Load audio (soundfile)
    2. Stereo handling
    3. DC offset removal
    4. Loudness normalization (LUFS)
    5. Resample to TARGET_SR
    6. Apply start_offset and max_duration (segmenting)
    7. Estimate tuning and correct to A=440
    """
    meta = MetaData()

    # A1–2: load and stereo handling
    y, sr = _load_audio(file_path, stereo_mode=stereo_mode)

    # Handle start_offset (in seconds)
    if start_offset and start_offset > 0.0:
        start_samples = int(start_offset * sr)
        if start_samples < len(y):
            y = y[start_samples:]
        else:
            # If offset beyond audio length, return tiny silence
            y = np.zeros(int(sr * 0.5), dtype=np.float32)

    # Optional duration limit
    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * sr)
        if y.shape[0] > max_samples:
            y = y[:max_samples]

    # A3: DC removal
    y = _remove_dc(y)

    # A4: Loudness normalization
    try:
        y, measured_lufs = _normalize_lufs(y, sr, TARGET_LUFS)
        meta.lufs = measured_lufs
    except Exception:
        # Fallback: simple peak norm
        peak = float(np.max(np.abs(y)) + 1e-9)
        y = (y / peak).astype(np.float32)
        meta.lufs = TARGET_LUFS

    # A5: Resample
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="kaiser_best")
        sr = TARGET_SR

    meta.sample_rate = sr

    # A6: tuning estimate & correction
    try:
        tuning_offset = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        tuning_offset = 0.0

    meta.tuning_offset = tuning_offset

    if abs(tuning_offset) > 0.01:
        # librosa.estimate_tuning gives semitone fraction
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-tuning_offset)

    return y, sr, meta

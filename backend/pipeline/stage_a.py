import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import warnings
import os
import scipy.signal
from typing import Tuple, Optional, Dict
from enum import Enum
import torch
from demucs import pretrained
from demucs.apply import apply_model

from .models import MetaData, AudioType, AudioQuality, Stem, StageAOutput
from .config import PIANO_61KEY_CONFIG, StageAConfig


# ------------------------------------------------------------
# Audio Type Detection
# ------------------------------------------------------------

def detect_audio_type(y: np.ndarray, sr: int) -> AudioType:
    """
    Heuristic classification: MONOPHONIC vs POLYPHONIC_DOMINANT vs POLYPHONIC.
    """
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness))

    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    mean_contrast = float(np.mean(contrast))

    if mean_flatness < 0.05 and mean_contrast > 20.0:
        return AudioType.MONOPHONIC
    elif mean_flatness < 0.2:
        return AudioType.POLYPHONIC_DOMINANT
    else:
        return AudioType.POLYPHONIC


def warped_linear_prediction(y: np.ndarray, sr: int, order: int = 16) -> np.ndarray:
    """
    Optional LPC whitening. Currently unused in WI, but kept for future experiments.
    """
    if len(y) < order + 1:
        return y
    a = librosa.lpc(y, order=order)
    y_white = scipy.signal.lfilter(a, [1], y)
    return y_white


# ------------------------------------------------------------
# Demucs (kept here but NOT used in Stage A per WI; Stage B should call this)
# ------------------------------------------------------------

def apply_demucs(audio: np.ndarray, sr: int, model_name: str = "htdemucs") -> Dict[str, np.ndarray]:
    """
    Apply Hybrid Demucs for source separation.
    NOTE: Per WI, separation belongs to Stage B. This helper is here for Stage B to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = pretrained.get_model(model_name)
        model.to(device)
    except Exception as e:
        warnings.warn(f"Failed to load Demucs model: {e}. Returning original mix as stems.")
        return {"vocals": audio, "bass": audio, "drums": audio, "other": audio}

    if audio.ndim == 1:
        audio_stereo = np.stack([audio, audio])
    elif audio.ndim == 2:
        if audio.shape[0] > 2:
            audio_stereo = audio.T
        else:
            audio_stereo = audio
    else:
        audio_stereo = np.stack([audio, audio])

    model_sr = model.samplerate
    if sr != model_sr:
        audio_stereo = librosa.resample(audio_stereo, orig_sr=sr, target_sr=model_sr)

    wav = torch.tensor(audio_stereo, dtype=torch.float32).to(device)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-9)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False
        )[0]

    sources = sources * ref.std() + ref.mean()
    stems = {}
    source_names = model.sources
    for i, name in enumerate(source_names):
        stem_audio = sources[i].cpu().numpy()
        stem_mono = np.mean(stem_audio, axis=0)
        stems[name] = stem_mono

    return stems


# ------------------------------------------------------------
# Stage A: Load + Conditioning
# ------------------------------------------------------------

def load_and_preprocess(
    audio_path: str,
    config: StageAConfig = PIANO_61KEY_CONFIG.stage_a,
    fast_mode: bool = False,
) -> StageAOutput:
    """
    Stage A: Signal Conditioning for 61-key piano (and general music).

    WI alignment:
    - Target sample rate (e.g., 22050 Hz).
    - Channel handling (mono sum).
    - DC offset removal.
    - Transient pre-emphasis (hammer attacks).
    - High-pass filter at ~55 Hz (4th order).
    - Silence trimming (top_db ~50).
    - Loudness normalization (EBU R128 → target LUFS).
    - Peak limiter (soft clip or -1 dB ceiling).
    - Noise floor estimation (30th percentile RMS).
    - BPM / beat grid detection.
    - Audio type + quality classification.
    - Returns a normalized mono "mix" stem and meta.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # -----------------------------
    # 1. Load with target sample rate
    # -----------------------------
    target_sr = config.target_sample_rate
    try:
        # librosa returns shape (n_channels, n_samples) when mono=False
        y, sr = librosa.load(audio_path, sr=target_sr, mono=False)
    except Exception:
        y, sr = sf.read(audio_path)
        if y.ndim == 2:
            y = y.T  # (channels, samples)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    # -----------------------------
    # 2. Channel Handling
    # -----------------------------
    n_channels = 1
    if y.ndim > 1:
        n_channels = y.shape[0]
        if config.channel_handling == "mono_sum":
            y = np.mean(y, axis=0)
        elif config.channel_handling == "left_only":
            y = y[0]
        elif config.channel_handling == "right_only":
            y = y[1]

    # Ensure 1D from here
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # -----------------------------
    # 3. DC Offset Removal
    # -----------------------------
    if config.dc_offset_removal:
        y = y - np.mean(y)

    # -----------------------------
    # 4. Transient Pre-emphasis (hammer attacks)
    # -----------------------------
    if config.transient_pre_emphasis.get("enabled", False):
        alpha = config.transient_pre_emphasis.get("alpha", 0.97)
        # y[t] = x[t] - alpha * x[t-1]
        y = scipy.signal.lfilter([1.0, -alpha], [1.0], y)

    # -----------------------------
    # 5. High-Pass Filter (Butterworth, 55Hz, 4th-order)
    # -----------------------------
    cutoff = config.high_pass_filter_cutoff.get("value", 55.0)
    order = config.high_pass_filter_order.get("value", 4)
    sos = scipy.signal.butter(order, cutoff, btype="hp", fs=sr, output="sos")
    y = scipy.signal.sosfilt(sos, y)

    # -----------------------------
    # 6. Silence Trim
    # -----------------------------
    if config.silence_trimming.get("enabled", True):
        top_db = config.silence_trimming.get("top_db", 50.0)
        y_trimmed, trim_idx = librosa.effects.trim(y, top_db=top_db)
        # Optional: track trim offsets in seconds
        trim_start_sec = float(trim_idx[0] / sr)
        trim_end_sec = float(trim_idx[1] / sr)
    else:
        y_trimmed = y
        trim_start_sec = 0.0
        trim_end_sec = float(len(y_trimmed) / sr)

    # -----------------------------
    # 7. Loudness Normalization (EBU R128)
    # -----------------------------
    norm_gain = 0.0
    y_norm = y_trimmed
    if config.loudness_normalization.get("enabled", True):
        target_lufs = float(config.loudness_normalization.get("target_lufs", -23.0))
        meter = pyln.Meter(sr)
        try:
            if len(y_trimmed) / sr >= 0.4:
                loudness = meter.integrated_loudness(y_trimmed)
                if not (np.isneginf(loudness) or loudness < -70.0):
                    y_norm = pyln.normalize.loudness(y_trimmed, loudness, target_lufs)
                    norm_gain = float(target_lufs - loudness)
            # If too short or too quiet, skip normalization
        except Exception as e:
            warnings.warn(f"Loudness normalization failed: {e}")
            y_norm = y_trimmed

    # 7b. Peak Limiter (optional, from config.peak_limiter)
    pl_conf = getattr(config, "peak_limiter", {"enabled": False})
    if pl_conf.get("enabled", False):
        mode = pl_conf.get("mode", "soft")        # "soft" or "hard"
        ceiling_db = pl_conf.get("ceiling_db", -1.0)
        # Convert dB ceiling to linear amplitude
        ceiling_lin = 10.0 ** (ceiling_db / 20.0)

        if mode == "soft":
            # Soft clipping using tanh
            # scale into [-1,1] around the ceiling, then tanh, then scale back
            y_norm = np.tanh(y_norm / ceiling_lin) * ceiling_lin
        else:
            # Hard clip
            y_norm = np.clip(y_norm, -ceiling_lin, ceiling_lin)

    # -----------------------------
    # 8. Peak Limiter (Soft clip / -1 dB ceiling)
    # -----------------------------
    if config.peak_limiter.get("enabled", False):
        mode = config.peak_limiter.get("mode", "ceiling_db")
        if mode == "tanh":
            # Soft limiting via tanh
            peak = np.max(np.abs(y_norm)) + 1e-9
            # Scale to roughly +/-1 before tanh
            y_scaled = y_norm / peak
            y_limited = np.tanh(y_scaled)
            # Rescale to preserve approximate RMS
            rms_orig = np.sqrt(np.mean(y_norm**2)) + 1e-9
            rms_lim = np.sqrt(np.mean(y_limited**2)) + 1e-9
            y_norm = y_limited * (rms_orig / rms_lim)
        elif mode == "ceiling_db":
            # Hard-ish ceiling at, e.g., -1 dBFS
            ceiling_db = float(config.peak_limiter.get("ceiling_db", -1.0))
            ceiling_lin = 10.0 ** (ceiling_db / 20.0)
            peak = np.max(np.abs(y_norm)) + 1e-9
            if peak > ceiling_lin:
                y_norm = y_norm * (ceiling_lin / peak)

    # -----------------------------
    # 9. Noise Floor Estimation (30th-percentile RMS)
    # -----------------------------
    percentile = config.noise_floor_estimation.get("percentile", 30)
    frame_len = 2048
    hop = 1024
    if len(y_norm) >= frame_len:
        rms = librosa.feature.rms(y=y_norm, frame_length=frame_len, hop_length=hop)[0]
        noise_floor_rms = float(np.percentile(rms, percentile))
        noise_floor_db = float(20.0 * np.log10(noise_floor_rms + 1e-9))
    else:
        noise_floor_rms = 0.0
        noise_floor_db = -80.0

    # -----------------------------
    # 10. BPM / Beat Grid
    # -----------------------------
    bpm = None
    beats = []
    if config.bpm_detection.get("enabled", True):
        try:
            bpm, beat_frames = librosa.beat.beat_track(y=y_norm, sr=sr)
            beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        except Exception:
            bpm = None
            beats = []

    # -----------------------------
    # 11. Audio Type & Quality
    # -----------------------------
    audio_type = detect_audio_type(y_norm, sr)
    ext = os.path.splitext(audio_path)[1].lower()
    quality = AudioQuality.LOSSY if ext in [".mp3", ".ogg", ".m4a"] else AudioQuality.LOSSLESS

    # -----------------------------
    # 12. Stems (Stage A only returns conditioned mix)
    # -----------------------------
    stems_output: Dict[str, Stem] = {}

    # Stage A returns normalized mix; Stage B will handle separation.
    stems_output["mix"] = Stem(audio=y_norm, sr=sr, type="mix")

    # For convenience, some pipelines may expect a 'vocals' or 'main' stem;
    # we can alias mix → vocals for monophonic / fast_mode.
    if fast_mode or audio_type == AudioType.MONOPHONIC:
        stems_output["vocals"] = Stem(audio=y_norm, sr=sr, type="vocals")

    # -----------------------------
    # 13. MetaData construction
    # -----------------------------
    # Global hop_length for F0 detectors; you may adjust to 256/512 consistently with Stage B.
    global_hop = getattr(config, "global_hop_length", 512)

    duration_sec = float(len(y_norm) / sr)
    rms_db = float(20.0 * np.log10(np.sqrt(np.mean(y_norm**2)) + 1e-9))

    meta = MetaData(
        original_sr=sr,
        target_sr=target_sr,
        sample_rate=target_sr,
        duration_sec=duration_sec,
        window_size=2048,
        hop_length=global_hop,
        tempo_bpm=bpm,
        lufs=config.loudness_normalization.get("target_lufs", -23.0),
        processing_mode=audio_type.value,
        audio_type=audio_type,
        audio_quality=quality,
        audio_path=audio_path,
        n_channels=n_channels,
        normalization_gain_db=norm_gain,
        rms_db=rms_db,
    )

    # Attach noise floor & beats if MetaData supports these fields
    try:
        meta.noise_floor_rms = noise_floor_rms
        meta.noise_floor_db = noise_floor_db
        meta.beats = beats
        meta.trim_start_sec = trim_start_sec
        meta.trim_end_sec = trim_end_sec
    except Exception:
        # If MetaData doesn't have these attributes yet, we just skip assignment.
        pass

    return StageAOutput(stems=stems_output, meta=meta, audio_type=audio_type)

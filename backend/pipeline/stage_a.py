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

def detect_audio_type(y: np.ndarray, sr: int) -> AudioType:
    """
    Heuristic to detect if audio is Monophonic or Polyphonic.
    """
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = np.mean(flatness)

    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    mean_contrast = np.mean(contrast)

    if mean_flatness < 0.05 and mean_contrast > 20.0:
        return AudioType.MONOPHONIC
    elif mean_flatness < 0.2:
        return AudioType.POLYPHONIC_DOMINANT
    else:
        return AudioType.POLYPHONIC

def warped_linear_prediction(y: np.ndarray, sr: int, order: int = 16) -> np.ndarray:
    """
    Standard LPC whitening as a proxy.
    """
    if len(y) < order + 1:
        return y
    a = librosa.lpc(y, order=order)
    y_white = scipy.signal.lfilter(a, [1], y)
    return y_white

def apply_demucs(audio: np.ndarray, sr: int, model_name: str = "htdemucs") -> Dict[str, np.ndarray]:
    """
    Apply Hybrid Demucs.
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

    model_sr = model.samplerate
    if sr != model_sr:
        audio_stereo = librosa.resample(audio_stereo, orig_sr=sr, target_sr=model_sr)

    wav = torch.tensor(audio_stereo, dtype=torch.float32).to(device)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    with torch.no_grad():
        sources = apply_model(model, wav[None], shifts=1, split=True, overlap=0.25, progress=False)[0]

    sources = sources * ref.std() + ref.mean()
    stems = {}
    source_names = model.sources
    for i, name in enumerate(source_names):
        stem_audio = sources[i].cpu().numpy()
        stem_mono = np.mean(stem_audio, axis=0)
        stems[name] = stem_mono

    return stems

def load_and_preprocess(
    audio_path: str,
    config: StageAConfig = PIANO_61KEY_CONFIG.stage_a,
    fast_mode: bool = False,
) -> StageAOutput:
    """
    Stage A: Signal Conditioning.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load with target sample rate
    target_sr = config.target_sample_rate
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=False)
    except Exception:
        y, sr = sf.read(audio_path)
        if y.ndim == 2: y = y.T
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    # 2. Channel Handling
    n_channels = 1
    if y.ndim > 1:
        n_channels = y.shape[0]
        if config.channel_handling == "mono_sum":
            y = np.mean(y, axis=0)
        elif config.channel_handling == "left_only":
            y = y[0]
        elif config.channel_handling == "right_only":
            y = y[1]

    # Ensure y is 1D from here
    if y.ndim > 1: y = np.mean(y, axis=0)

    # 3. DC Offset Removal
    if config.dc_offset_removal:
        y = y - np.mean(y)

    # 4. Transient Pre-emphasis
    if config.transient_pre_emphasis.get("enabled", False):
        alpha = config.transient_pre_emphasis.get("alpha", 0.97)
        # y[t] = x[t] - alpha * x[t-1]
        y = scipy.signal.lfilter([1, -alpha], [1], y)

    # 5. High-Pass Filter (Butterworth)
    # Cutoff: 55Hz, Order: 4
    cutoff = config.high_pass_filter_cutoff.get("value", 55.0)
    order = config.high_pass_filter_order.get("value", 4)
    sos = scipy.signal.butter(order, cutoff, btype='hp', fs=sr, output='sos')
    y = scipy.signal.sosfilt(sos, y)

    # 6. Silence Trim
    if config.silence_trimming.get("enabled", True):
        top_db = config.silence_trimming.get("top_db", 50)
        y_trimmed, trim_idx = librosa.effects.trim(y, top_db=top_db)
    else:
        y_trimmed = y

    # 7. Loudness Normalization (EBU R128)
    norm_gain = 0.0
    if config.loudness_normalization.get("enabled", True):
        target_lufs = config.loudness_normalization.get("target_lufs", -23.0)
        meter = pyln.Meter(sr)
        try:
            if len(y_trimmed) / sr >= 0.4:
                loudness = meter.integrated_loudness(y_trimmed)
                if not (np.isneginf(loudness) or loudness < -70.0):
                    y_norm = pyln.normalize.loudness(y_trimmed, loudness, target_lufs)
                    norm_gain = target_lufs - loudness
                else:
                    y_norm = y_trimmed
            else:
                y_norm = y_trimmed
        except Exception as e:
            warnings.warn(f"Loudness norm failed: {e}")
            y_norm = y_trimmed
    else:
        y_norm = y_trimmed

    # 8. Noise Floor Estimation
    percentile = config.noise_floor_estimation.get("percentile", 30)
    # Frame RMS
    frame_len = 2048
    hop = 1024
    if len(y_norm) >= frame_len:
        rms = librosa.feature.rms(y=y_norm, frame_length=frame_len, hop_length=hop)[0]
        noise_floor_rms = np.percentile(rms, percentile)
        noise_floor_db = 20 * np.log10(noise_floor_rms + 1e-9)
    else:
        noise_floor_rms = 0.0
        noise_floor_db = -80.0

    # 9. BPM / Beat Grid
    bpm = None
    beats = []
    if config.bpm_detection.get("enabled", True):
        try:
            bpm, beat_frames = librosa.beat.beat_track(y=y_norm, sr=sr)
            beats = librosa.frames_to_time(beat_frames, sr=sr)
        except Exception:
            pass

    # 10. Classification & Separation Preparation
    audio_type = detect_audio_type(y_norm, sr)
    ext = os.path.splitext(audio_path)[1].lower()
    quality = AudioQuality.LOSSY if ext in ['.mp3', '.ogg', '.m4a'] else AudioQuality.LOSSLESS

    stems_output = {}

    if fast_mode or audio_type == AudioType.MONOPHONIC:
        # Skip Demucs
        stems_output["mix"] = Stem(audio=y_norm, sr=sr, type="mix")
        # Populate others as needed or just use mix
        stems_output["vocals"] = Stem(audio=y_norm, sr=sr, type="vocals") # Treat mix as main stem
    else:
        # Run Demucs on the PRE-PROCESSED mono (or stereo if we kept it)
        # WI says "Load with sr...", implies processing on y.
        # Demucs usually takes stereo. But we did mono sum.
        # We can feed mono to Demucs (duplicated).
        # Should we use the TRIMMED/NORMALIZED y? Yes.

        # NOTE: Demucs might work better on original raw audio, but WI implies linear flow.
        # "Normalize signal... so that all detectors downstream behave predictably"
        # Since Separation is Stage B in WI?
        # WI says "2.1 Stage A ... Return y (processed)".
        # "2.2 Stage B ... Source Separation".
        # So Separation logic should be in Stage B, but typically we do it before detectors.
        # My plan said "Refactor Stage A ... Return StageAOutput".
        # Current Stage A included Separation.
        # WI structure: "2.1 Stage A" returns y. "2.2 Stage B" does separation.
        # I will move separation to Stage B strictly.

        # So Stage A just returns the conditioned mix.
        stems_output["mix"] = Stem(audio=y_norm, sr=sr, type="mix")

    meta = MetaData(
        original_sr=sr,
        target_sr=target_sr,
        sample_rate=target_sr,
        duration_sec=len(y_norm) / sr,
        window_size=2048,
        hop_length=512, # Standard? WI mentions 160 for vocals, detector specific.
                        # Global hop?
                        # WI: "All detectors must produce F0 at the same hop size".
                        # Let's set a global hop, e.g. 256 or 512.
        tempo_bpm=bpm,
        lufs=config.loudness_normalization.get("target_lufs", -23),
        processing_mode=audio_type.value,
        audio_type=audio_type,
        audio_quality=quality,
        audio_path=audio_path,
        n_channels=n_channels,
        normalization_gain_db=norm_gain,
        rms_db=20 * np.log10(np.sqrt(np.mean(y_norm**2))) if len(y_norm) > 0 else -80
    )

    # Store noise floor in meta for Stage C/D use? Or just in analysis data.
    # MetaData doesn't have noise_floor field.
    # But Stage A returns it.

    # I will attach beat times to analysis object later, or put in MetaData if I add a field.
    # MetaData has `beats` (added in Models).

    return StageAOutput(stems=stems_output, meta=meta, audio_type=audio_type)

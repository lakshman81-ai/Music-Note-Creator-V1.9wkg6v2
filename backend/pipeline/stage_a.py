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
from .models import MetaData, AudioType, Stem, StageAOutput

# Constants
TARGET_LUFS = -23.0  # EBU R128 standard
MIN_DURATION_SEC = 0.120 # 120 ms
MAX_DURATION_SEC = 600.0
SILENCE_THRESHOLD_DB = 40  # top_db for trim

# Demucs Model
DEMUCS_MODEL_NAME = 'htdemucs'

def detect_audio_type(y: np.ndarray, sr: int) -> AudioType:
    """
    Heuristic to detect if audio is Monophonic or Polyphonic.
    Uses spectral flatness and crest factor.
    """
    # 1. Compute Spectral Flatness
    # High flatness -> Noise-like or complex polyphony
    # Low flatness -> Tonal (sine wave)
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = np.mean(flatness)

    # 2. Spectral Crest Factor (Peakiness)
    # Monophonic signals often have distinct harmonic peaks (high crest factor).
    # Polyphonic signals have denser spectra (lower crest factor).
    # We approximate this via standard deviation of magnitude spectrum.
    S = np.abs(librosa.stft(y))
    # Simple proxy: sparsity or contrast.
    # Let's use spectral contrast
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    mean_contrast = np.mean(contrast)

    # Heuristic Thresholds (Tuned loosely for now)
    # Monophonic usually has high contrast (clear peaks vs valleys) and low flatness.
    if mean_flatness < 0.05 and mean_contrast > 20.0:
        return AudioType.MONOPHONIC
    elif mean_flatness < 0.2:
        return AudioType.POLYPHONIC_DOMINANT
    else:
        return AudioType.POLYPHONIC

def warped_linear_prediction(y: np.ndarray, sr: int, order: int = 16, warping: float = 0.7) -> np.ndarray:
    """
    Applies Warped Linear Prediction (WLP) or standard LPC-based whitening.
    Report calls for WLP, but standard LPC is a good approximation if warping lib is missing.
    We implement standard LPC-based whitening as a robust fallback for "spectral whitening".
    """
    # Standard LPC whitening
    # 1. Calculate LPC coefficients
    if len(y) < order + 1:
        return y

    a = librosa.lpc(y, order=order)

    # 2. Inverse filter (Whitening)
    # The residue of the prediction filter is the whitened signal
    y_white = scipy.signal.lfilter(a, [1], y)

    return y_white

def apply_demucs(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    Apply Hybrid Demucs to separate audio into 4 stems: vocals, bass, drums, other.
    Input: (samples,) mono or stereo.
    Output: Dict with keys 'vocals', 'bass', 'drums', 'other'.
    """
    # Prepare model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = pretrained.get_model(DEMUCS_MODEL_NAME)
        model.to(device)
    except Exception as e:
        warnings.warn(f"Failed to load Demucs model: {e}. Returning original mix as stems.")
        return {"vocals": audio, "bass": audio, "drums": audio, "other": audio}

    # Prepare input tensor: (Batch, Channels, Time)
    # Demucs expects stereo usually. If mono, duplicate channels.
    if audio.ndim == 1:
        audio_stereo = np.stack([audio, audio])
    elif audio.ndim == 2:
        if audio.shape[0] > 2: # (Time, Channels) -> (Channels, Time)
             audio_stereo = audio.T
        else:
             audio_stereo = audio

    # Resample to model samplerate if needed (Demucs handles this usually, but good to be safe)
    # htdemucs is usually 44.1k
    model_sr = model.samplerate
    if sr != model_sr:
        audio_stereo = librosa.resample(audio_stereo, orig_sr=sr, target_sr=model_sr)

    wav = torch.tensor(audio_stereo, dtype=torch.float32).to(device)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    # Inference
    with torch.no_grad():
        # shifts=1, split=True for memory efficiency
        sources = apply_model(model, wav[None], shifts=1, split=True, overlap=0.25, progress=False)[0]

    # De-normalize
    sources = sources * ref.std() + ref.mean()

    # Extract stems
    stems = {}
    source_names = model.sources
    for i, name in enumerate(source_names):
        # Convert back to mono for downstream processing (as requested by SwiftF0/SACF)
        # Average the stereo channels
        stem_audio = sources[i].cpu().numpy()
        stem_mono = np.mean(stem_audio, axis=0)
        stems[name] = stem_mono

    return stems

def load_and_preprocess(
    audio_path: str,
    target_sr: int = 44100, # Defaulting to high quality as per report for Autocorr
    trim_silence: bool = True,
) -> StageAOutput:
    """
    Stage A: Robust audio loading, silence trimming, mono conversion, resampling,
    EBU R128 normalization, Audio Type Classification, and Source Separation.
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

    # Duration Check
    duration_sec = float(y_orig.shape[-1]) / sr_orig
    if duration_sec > MAX_DURATION_SEC:
        warnings.warn(f"Audio too long: {duration_sec:.2f}s. Trimming to {MAX_DURATION_SEC}s.")
        limit_samples = int(MAX_DURATION_SEC * sr_orig)
        if y_orig.ndim > 1:
            y_orig = y_orig[:, :limit_samples]
        else:
            y_orig = y_orig[:limit_samples]
        duration_sec = MAX_DURATION_SEC

    # 2. Downmix to mono for initial analysis and SwiftF0
    n_channels = 1
    if y_orig.ndim > 1:
        n_channels = y_orig.shape[0]
        y_mono = np.mean(y_orig, axis=0)
    else:
        y_mono = y_orig

    # 3. Silence Trimming
    if trim_silence:
        y_trimmed, trim_idx = librosa.effects.trim(y_mono, top_db=SILENCE_THRESHOLD_DB)
        # Apply trim to original stereo if needed, but for now we work with mono mostly.
        # If we need stereo for Demucs, we should trim stereo too.
        if y_orig.ndim > 1:
             # Map trim indices
             y_orig_trimmed = y_orig[:, trim_idx[0]:trim_idx[1]]
        else:
             y_orig_trimmed = y_orig[trim_idx[0]:trim_idx[1]]
    else:
        y_trimmed = y_mono
        y_orig_trimmed = y_orig

    # 4. Normalization (EBU R128)
    # Normalize the mono signal (used for analysis).
    # For Demucs, we feed the raw (trimmed) audio and it handles norm internally,
    # but let's normalize y_trimmed for consisten AudioType detection.
    meter = pyln.Meter(sr_orig)
    loudness = meter.integrated_loudness(y_trimmed)
    if not (np.isneginf(loudness) or loudness < -70.0):
        y_norm = pyln.normalize.loudness(y_trimmed, loudness, TARGET_LUFS)
        norm_gain = TARGET_LUFS - loudness
    else:
        y_norm = y_trimmed
        norm_gain = 0.0

    # 5. Audio Type Detection
    audio_type = detect_audio_type(y_norm, sr_orig)
    print(f"Detected Audio Type: {audio_type}")

    stems_output = {}

    # 6. Branching Logic
    if audio_type == AudioType.MONOPHONIC:
        # Path A: Monophonic
        # Resample to 16kHz for SwiftF0
        if sr_orig != 16000:
            y_16k = librosa.resample(y_norm, orig_sr=sr_orig, target_sr=16000)
        else:
            y_16k = y_norm

        stems_output["vocals"] = Stem(audio=y_16k, sr=16000, name="vocals")
        # Provide others as empty or copies?
        # Ideally, we just provide the main one.

    else:
        # Path B: Polyphonic (Demucs)
        print("Starting Source Separation (Demucs)...")
        # Use y_orig_trimmed (stereo if avail) for Demucs for best results
        demucs_stems = apply_demucs(y_orig_trimmed, sr_orig)

        # Process Stems
        # Vocals & Bass -> 16kHz (for SwiftF0)
        for name in ["vocals", "bass"]:
            s = demucs_stems[name]
            s_16k = librosa.resample(s, orig_sr=44100, target_sr=16000) # Demucs usually outputs 44.1k
            stems_output[name] = Stem(audio=s_16k, sr=16000, name=name)

        # Other -> 44.1kHz + Whitening (for SACF)
        other = demucs_stems["other"]
        # Whitening
        other_white = warped_linear_prediction(other, sr=44100)
        stems_output["other"] = Stem(audio=other_white, sr=44100, name="other")

        # Drums -> Keep raw or discard? Keep for reference.
        stems_output["drums"] = Stem(audio=demucs_stems["drums"], sr=44100, name="drums")

    # Meta Data Construction
    meta = MetaData(
        original_sr=sr_orig,
        target_sr=target_sr,
        sample_rate=target_sr, # Global target
        duration_sec=duration_sec,
        window_size=1024,
        hop_length=256,
        time_signature="4/4",
        tempo_bpm=None,
        lufs=TARGET_LUFS,
        processing_mode=audio_type.value,
        audio_type=audio_type,
        audio_path=audio_path,
        n_channels=n_channels,
        normalization_gain_db=norm_gain,
        rms_db=20 * np.log10(np.sqrt(np.mean(y_norm**2))) if len(y_norm) > 0 else -80
    )

    return StageAOutput(stems=stems_output, meta=meta, audio_type=audio_type)

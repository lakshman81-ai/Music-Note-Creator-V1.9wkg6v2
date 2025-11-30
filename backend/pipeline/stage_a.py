import soundfile as sf
import numpy as np
import pyloudnorm as pyln
import librosa
from backend.pipeline.models import MetaData

def load_and_preprocess(file_path: str, stereo_mode: bool, start_offset: float = 0.0, max_duration: float = None) -> tuple[np.ndarray, int, MetaData]:
    """
    Stage A: Advanced Pre-processing

    1. Input Loading (soundfile)
    2. Stereo Handling (Mixdown vs Mid/Side)
    3. DC Offset Removal
    4. Loudness Normalization (LUFS)
    5. Anti-Aliasing Resampling
    6. Adaptive Windowing (returned as metadata or handled in Stage B, here we define config)
    7. SNR Gate
    8. Automatic Tuning
    """

    meta = MetaData()
    meta.processing_mode = "stereo" if stereo_mode else "mono"

    # A1. Input Loading
    # Use soundfile for precise channel metadata
    # Load raw first to check channels
    try:
        # We assume file_path is valid.
        # soundfile.read returns (data, samplerate)
        # data is (frames, channels)

        # Handle start_offset and max_duration manually or via read arguments?
        # sf.read supports 'start' and 'stop' frames. But we usually think in seconds.
        # Let's inspect first.
        info = sf.info(file_path)
        sr_native = info.samplerate

        start_frame = int(start_offset * sr_native)
        frames_to_read = -1
        if max_duration:
            frames_to_read = int(max_duration * sr_native)

        # Safety check for start_frame
        if start_frame >= info.frames:
            raise ValueError("Start offset is beyond file duration.")

        y_native, sr = sf.read(file_path, start=start_frame, frames=frames_to_read, always_2d=True)

        # y_native is (samples, channels)
    except Exception as e:
        print(f"Error loading file: {e}")
        # Fallback to librosa if soundfile fails (e.g. some mp3 codecs)
        y_native, sr = librosa.load(file_path, sr=None, mono=False, offset=start_offset, duration=max_duration)
        if y_native.ndim == 1:
            y_native = y_native[:, np.newaxis] # make (samples, 1)
        else:
            y_native = y_native.T # make (samples, channels)

    # A2. Stereo Handling
    # Shape of y_native is (samples, channels)
    num_channels = y_native.shape[1]

    if num_channels == 1:
        y_processed = y_native[:, 0]
    else:
        # Stereo input
        left = y_native[:, 0]
        right = y_native[:, 1]

        if not stereo_mode:
            # A2.1: Mixdown uniformly
            y_processed = np.mean(y_native, axis=1)
        else:
            # A2.2: Intelligent Stereo Analysis
            # "Analyze: Left, Right, Mid, Side"
            mid = (left + right) / 2
            side = (left - right) / 2

            # "Detect Phase cancellation, Stereo imbalance"
            # Simple heuristic: Check energy of Mid vs Side
            e_mid = np.sum(mid**2)
            e_side = np.sum(side**2)

            # Heuristic from specs:
            # "If melodic component stronger in L/R -> use per-channel" (Not fully implemented in single-stream output context)
            # "If centered -> use Mid"
            # "If phase-distorted -> fallback to mono"

            # For this pipeline, we need to produce a single 'y' for pitch detection (unless we run multi-channel transcription)
            # The specs imply "Fuse results per-engine rules".
            # To simplify for the single 'y' return, we will default to Mid unless Side is very dominant (phase issues).

            if e_mid < 0.1 * e_side:
                 # Phase cancellation likely, Mid is weak.
                 # Fallback to mono mix of absolute values? Or just left?
                 # Spec says: "If phase-distorted -> fallback to mono".
                 # Usually mono mix (L+R)/2 cancels out if phase distorted.
                 # So we might want (L - R)/2 (Side) or just one channel.
                 # Let's use one channel to be safe, or stick to mean if the user insists on "mono".
                 # But the spec says "fallback to mono", which usually implies (L+R)/2, which is 'Mid'.
                 # If Mid is zero, that's bad.
                 y_processed = left # Arbitrary fallback to avoid silence
            else:
                 y_processed = mid

    # A3. DC Offset Removal
    # "Mandatory: y = y - np.mean(y)"
    y_processed = y_processed - np.mean(y_processed)

    # A4. Loudness Normalization (LUFS)
    # Target: -14 LUFS Integrated
    try:
        meter = pyln.Meter(sr) # create BS.1770 meter
        loudness = meter.integrated_loudness(y_processed)
        meta.lufs = float(loudness)

        # normalize to -14 LUFS
        target_lufs = -14.0
        y_processed = pyln.normalize.loudness(y_processed, loudness, target_lufs)
    except Exception as e:
        print(f"LUFS normalization warning: {e}. Fallback to peak norm.")
        # Fallback: Peak Normalize
        peak = np.max(np.abs(y_processed))
        if peak > 0:
            y_processed = y_processed / peak * 0.9 # -1 dB ish

    # A5. Anti-Aliasing Resampling
    # Target 22050 or 44100.
    target_sr = 22050
    if sr != target_sr:
        # librosa.resample uses kaiser_best by default in older versions, or soxr in newer.
        # Spec says: res_type="kaiser_best"
        y_processed = librosa.resample(y_processed, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr

    # A6. Adaptive Windowing
    # "Decision based on spectral centroid"
    centroid = librosa.feature.spectral_centroid(y=y_processed, sr=sr).mean()
    # "Low-pitched audio -> larger window (4096), High/Percussive -> shorter (1024)"
    # Threshold logic:
    # A4 (440Hz) is approx center.
    if centroid < 250: # Low freq average
        window_size = 4096
    else:
        window_size = 1024

    meta.window_size = window_size

    # A7. SNR Gate
    # SNR = 10 * log10(signal_power / noise_floor)
    # Estimate noise floor?
    # Simple approach: assume lowest 5% energy frames are noise.
    S = np.abs(librosa.stft(y_processed))
    rms = librosa.feature.rms(S=S)[0]
    signal_power = np.mean(rms**2)
    # Estimate noise as mean of the bottom 10th percentile of RMS
    noise_power = np.mean(np.sort(rms)[:int(len(rms)*0.1)]**2)

    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = 100 # Infinity

    meta.snr = float(snr)

    if snr < 10: # Threshold example
        print("Warning: Low SNR detected.")
        # "Apply spectral noise reduction" - omitted for brevity/complexity in this pass,
        # or we could use a simple gate.

    # A8. Automatic Tuning / A-Reference Detection
    # estimate_tuning returns offset in bins (fraction of semitone)
    tuning_offset = librosa.estimate_tuning(y=y_processed, sr=sr)
    meta.tuning_offset = float(tuning_offset) # This is in fractions of a bin (semitone)

    # "Apply pitch shift: steps = -tuning_offset_cents/100"
    # librosa.estimate_tuning returns fraction of semitone (e.g., -0.1 means -10 cents)
    # So we shift by -tuning_offset
    if abs(tuning_offset) > 0.01:
        y_processed = librosa.effects.pitch_shift(y_processed, sr=sr, n_steps=-tuning_offset)

    return y_processed, sr, meta

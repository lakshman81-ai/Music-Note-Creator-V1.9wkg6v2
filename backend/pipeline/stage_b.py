from typing import List, Tuple, Optional, Dict
import numpy as np
import librosa
import scipy.signal
from .models import MetaData, FramePitch, NoteEvent, ChordEvent, StageAOutput, Stem
from .detectors import YinDetector, CQTDetector, SACFDetector, SwiftF0Detector

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

def create_harmonic_mask(stft: np.ndarray, f0_curve: np.ndarray, sr: int, width: float = 0.03) -> np.ndarray:
    """
    Creates a spectral mask to suppress harmonics of the given pitch trajectory.
    width: Fractional bandwidth to mask around harmonics (default 3%).
    """
    n_freqs, n_frames = stft.shape
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_freqs - 1) * 2)
    mask = np.zeros_like(stft, dtype=np.float32)

    # We can optimize this using broadcasting or process frame-by-frame
    # Frame-by-frame is easier to read and implement correctly for varying f0

    for t in range(min(n_frames, len(f0_curve))):
        f0 = f0_curve[t]
        if f0 <= 0:
            continue

        # Identify harmonics up to Nyquist
        harmonics = np.arange(1, 20) * f0
        harmonics = harmonics[harmonics < sr / 2]

        for h_freq in harmonics:
            # Bandwidth
            bw = h_freq * width
            f_low = h_freq - bw/2
            f_high = h_freq + bw/2

            # Find bins
            # np.searchsorted is fast
            idx_start = np.searchsorted(fft_freqs, f_low)
            idx_end = np.searchsorted(fft_freqs, f_high)

            # Apply mask (1.0 means fully masked/removed later)
            mask[idx_start:idx_end, t] = 1.0

    return mask

def iterative_spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    detector,
    max_polyphony: int = 4,
    termination_conf: float = 0.15
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Performs ISS to extract multiple pitch trajectories.
    Returns list of (f0_curve, conf_curve).
    """
    extracted = []
    y_resid = audio.copy()

    # Pre-calculate STFT parameters to ensure consistency
    n_fft = 2048
    hop_length = detector.hop_length

    for i in range(max_polyphony):
        # 1. Detect Pitch
        f0, conf = detector.predict(y_resid)

        # Check termination (if mean confidence is too low)
        # Or if no voiced frames
        if np.mean(conf) < termination_conf and np.max(conf) < termination_conf * 1.5:
            break

        extracted.append((f0, conf))

        # 2. Spectral Subtraction
        # Compute STFT
        S = librosa.stft(y_resid, n_fft=n_fft, hop_length=hop_length)

        # Create Mask
        mask = create_harmonic_mask(S, f0, sr)

        # Apply Soft Mask (Inverse)
        # 1 - mask (where mask is 1, we multiply by 0)
        # Smooth mask? Hard mask is prone to artifacts, but sufficient for subtraction.
        S_clean = S * (1 - mask)

        # Reconstruct
        y_resid = librosa.istft(S_clean, hop_length=hop_length, length=len(y_resid))

    return extracted

def extract_features(
    stage_a_output: StageAOutput,
    use_crepe: bool = False, # Deprecated/Ignored
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]]:
    """
    Stage B: Adaptive Pitch Detection (Stem-based).
    """
    meta = stage_a_output.meta
    stems = stage_a_output.stems
    sr = meta.sample_rate # Target SR
    hop_length = meta.hop_length

    # Initialize Detectors
    # Note: Detectors need to match the SR of the stem provided by Stage A
    # Stage A output stems have their own .sr property.

    # Results container: Map stem name to list of (f0, conf) tracks
    stem_results: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    # 1. Vocals & Bass (SwiftF0)
    # We use the mono detector
    # SwiftF0 expects 16kHz usually, Stage A provides 16kHz for these.

    for name in ["vocals", "bass"]:
        if name in stems:
            stem = stems[name]
            # Init detector with stem's SR
            det = SwiftF0Detector(stem.sr, hop_length, fmin=40.0, fmax=2000.0) # Tuned range
            f0, conf = det.predict(stem.audio)

            # Add to results (single track)
            stem_results[name] = [(f0, conf)]

    # 2. Other (SACF + ISS)
    if "other" in stems:
        stem = stems["other"]
        # Init detector
        # SACF benefits from higher SR (44.1k), which Stage A provides.
        det = SACFDetector(stem.sr, hop_length, fmin=60.0, fmax=2000.0)

        # Run ISS
        # Max polyphony 4-6
        extracted_tracks = iterative_spectral_subtraction(stem.audio, stem.sr, det, max_polyphony=4)
        stem_results["other"] = extracted_tracks

    # 3. Merge into Timeline
    # We need to align all tracks to the same time grid.
    # Assuming hop_length is consistent, the frames should align.
    # However, different SRs might result in slightly different frame counts if not handled carefully.
    # Stage A ensures consistent duration?
    # librosa.stft depends on SR and hop_length.
    # If hop_length is 256 @ 44.1k vs 256 @ 16k, the time steps are different!
    # AHH. `hop_length` in `MetaData` is usually for the *analysis* target SR.
    # The detectors took `hop_length` in their constructor.
    # If SwiftF0 runs at 16k with hop 256, time step is 256/16000 = 16ms.
    # If SACF runs at 44.1k with hop 256, time step is 256/44100 = 5.8ms.
    # We must synchronize them to the global analysis grid (meta.sample_rate).

    # Global Time Grid
    # We'll rely on timestamps.
    duration = meta.duration_sec
    # Or just use the longest track as reference?

    # Let's verify what `meta.hop_length` means.
    # In `stage_a.py`, `meta` is created with `sample_rate=target_sr`.
    # But stems have different SRs.
    # We should resample the pitch tracks to the common timeline.

    # Common Timeline: Defined by meta.sample_rate and meta.hop_length (e.g. 22050 / 512 ~ 23ms)
    # Or just choose one.

    # Let's define the target grid.
    target_sr = meta.sample_rate
    target_hop = meta.hop_length
    n_frames_global = int(duration * target_sr / target_hop) + 1
    times_global = librosa.frames_to_time(np.arange(n_frames_global), sr=target_sr, hop_length=target_hop)

    timeline: List[FramePitch] = []

    # Helper to resample track
    def resample_track(f0, conf, src_sr, src_hop):
        # Current times
        n = len(f0)
        times_src = librosa.frames_to_time(np.arange(n), sr=src_sr, hop_length=src_hop)

        # Interpolate to global times
        # Use simple interp1d
        if n < 2: return np.zeros(n_frames_global), np.zeros(n_frames_global)

        # Fill value 0
        f0_interp = np.interp(times_global, times_src, f0, left=0, right=0)
        conf_interp = np.interp(times_global, times_src, conf, left=0, right=0)
        return f0_interp, conf_interp

    # Collect all pitches per frame
    frame_pitches_map = [[] for _ in range(n_frames_global)] # List of lists of (pitch, conf)

    for name, tracks in stem_results.items():
        if name not in stems: continue
        stem_sr = stems[name].sr
        stem_hop = hop_length # We passed the same hop_length to detectors?
        # Detectors were init with `meta.hop_length`.
        # If SR differs, `hop_length` samples represents different time.
        # So we MUST resample.

        for f0, conf in tracks:
            f0_res, conf_res = resample_track(f0, conf, stem_sr, hop_length)

            for i in range(n_frames_global):
                if f0_res[i] > 10.0: # Filter noise
                    frame_pitches_map[i].append((f0_res[i], conf_res[i]))

    # Compute RMS for global timeline (using mix or just skipped)
    # Ideally we use the sum of stems or the original mix if available.
    # We can approximate by summing stem RMS? Or just 0.
    # Let's skip expensive RMS recalc or use a proxy.
    global_rms = np.zeros(n_frames_global)

    for i in range(n_frames_global):
        active = frame_pitches_map[i]

        # Sort by confidence
        active.sort(key=lambda x: x[1], reverse=True)

        # Determine dominant pitch (e.g. vocals > bass > other)
        # For now just max confidence
        dom_pitch = 0.0
        dom_conf = 0.0
        dom_midi = None

        if active:
            dom_pitch = active[0][0]
            dom_conf = active[0][1]
            dom_midi = int(round(hz_to_midi(dom_pitch)))

        fp = FramePitch(
            time=times_global[i],
            pitch_hz=dom_pitch,
            midi=dom_midi,
            confidence=dom_conf,
            rms=global_rms[i],
            active_pitches=active
        )
        timeline.append(fp)

    # 4. Smoothing
    # Apply median smoothing to the dominant pitch track?
    # And maybe to active pitches?
    # For now, let's smooth the dominant track for legacy compatibility.
    # Complex polyphonic smoothing is Stage C's job.

    dom_pitches = np.array([fp.pitch_hz for fp in timeline])
    smoothed = scipy.signal.medfilt(dom_pitches, kernel_size=5)

    for i in range(len(timeline)):
        if smoothed[i] > 0:
            timeline[i].pitch_hz = float(smoothed[i])
            timeline[i].midi = int(round(hz_to_midi(smoothed[i])))
        else:
            # If smoothed to 0, but we had something?
            # Median filter suppresses outliers.
            timeline[i].pitch_hz = 0.0
            timeline[i].midi = None

    return timeline, [], []

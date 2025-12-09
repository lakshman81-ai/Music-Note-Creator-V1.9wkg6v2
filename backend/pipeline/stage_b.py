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

    # 3. Merge into Timeline & Create Stem Timelines
    # We need to align all tracks to the same time grid.

    # Let's define the target grid based on global meta.
    target_sr = meta.sample_rate
    target_hop = meta.hop_length
    n_frames_global = int(duration * target_sr / target_hop) + 1
    times_global = librosa.frames_to_time(np.arange(n_frames_global), sr=target_sr, hop_length=target_hop)

    timeline: List[FramePitch] = []

    # Output structure for stem-specific timelines
    # We map "stem_name" -> List[FramePitch] (where each frame has 1 pitch usually, or multiple for 'other')
    # Actually, FramePitch is convenient.
    stem_timelines: Dict[str, List[FramePitch]] = {}

    # Helper to resample track
    def resample_track(f0, conf, src_sr, src_hop):
        n = len(f0)
        times_src = librosa.frames_to_time(np.arange(n), sr=src_sr, hop_length=src_hop)

        if n < 2: return np.zeros(n_frames_global), np.zeros(n_frames_global)

        f0_interp = np.interp(times_global, times_src, f0, left=0, right=0)
        conf_interp = np.interp(times_global, times_src, conf, left=0, right=0)
        return f0_interp, conf_interp

    # Collect all pitches per frame for the global merged view
    frame_pitches_map = [[] for _ in range(n_frames_global)]

    for name, tracks in stem_results.items():
        if name not in stems: continue
        stem_sr = stems[name].sr
        stem_hop = hop_length

        # Initialize stem timeline list
        stem_timelines[name] = []

        # To populate stem_timelines[name], we need to combine the tracks for this stem.
        # For 'vocals'/'bass', usually 1 track. For 'other', multiple tracks.
        # We need a per-frame list for this stem.
        stem_frame_pitches = [[] for _ in range(n_frames_global)]

        for f0, conf in tracks:
            f0_res, conf_res = resample_track(f0, conf, stem_sr, hop_length)

            for i in range(n_frames_global):
                if f0_res[i] > 10.0:
                    # Add to global mix
                    frame_pitches_map[i].append((f0_res[i], conf_res[i]))
                    # Add to stem specific mix
                    stem_frame_pitches[i].append((f0_res[i], conf_res[i]))

        # Create FramePitch objects for this stem
        for i in range(n_frames_global):
            active = stem_frame_pitches[i]
            # Dominant for this stem
            active.sort(key=lambda x: x[1], reverse=True)

            s_pitch = active[0][0] if active else 0.0
            s_conf = active[0][1] if active else 0.0
            s_midi = int(round(hz_to_midi(s_pitch))) if s_pitch > 0 else None

            fp = FramePitch(
                time=times_global[i],
                pitch_hz=s_pitch,
                midi=s_midi,
                confidence=s_conf,
                rms=0.0, # RMS not calculated yet
                active_pitches=active
            )
            stem_timelines[name].append(fp)

    # Build Global Timeline (Legacy / Merged View)
    global_rms = np.zeros(n_frames_global) # Placeholder

    for i in range(n_frames_global):
        active = frame_pitches_map[i]
        active.sort(key=lambda x: x[1], reverse=True)

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

    # 4. Smoothing (Global Timeline only)
    dom_pitches = np.array([fp.pitch_hz for fp in timeline])
    smoothed = scipy.signal.medfilt(dom_pitches, kernel_size=5)

    for i in range(len(timeline)):
        if smoothed[i] > 0:
            timeline[i].pitch_hz = float(smoothed[i])
            timeline[i].midi = int(round(hz_to_midi(smoothed[i])))
        else:
            timeline[i].pitch_hz = 0.0
            timeline[i].midi = None

    # Return structure needs to allow passing stem_timelines.
    # The signature is Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]].
    # We can't change signature easily without breaking callers if we don't control them.
    # But we control `transcription.py`.
    # However, to avoid changing signature too much, we will update `AnalysisData` in the caller.
    # For now, return timeline. The caller `transcription.py` needs to change to get `stem_timelines`.
    # To do that, we should probably change the return type to include stem_timelines.

    return timeline, [], [], stem_timelines

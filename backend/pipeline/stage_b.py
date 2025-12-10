from typing import List, Tuple, Optional, Dict
import numpy as np
import librosa
import scipy.signal
import scipy.interpolate
from .models import MetaData, FramePitch, NoteEvent, ChordEvent, StageAOutput, Stem, AudioQuality
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

    n_fft = 2048
    hop_length = detector.hop_length

    for i in range(max_polyphony):
        # 1. Detect Pitch
        f0, conf = detector.predict(y_resid)

        if np.mean(conf) < termination_conf and np.max(conf) < termination_conf * 1.5:
            break

        extracted.append((f0, conf))

        # 2. Spectral Subtraction
        S = librosa.stft(y_resid, n_fft=n_fft, hop_length=hop_length)
        mask = create_harmonic_mask(S, f0, sr)
        S_clean = S * (1 - mask)
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
    stem_results: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    # 1. Vocals & Bass (SwiftF0)
    for name in ["vocals", "bass"]:
        if name in stems:
            stem = stems[name]
            det = SwiftF0Detector(stem.sr, hop_length, fmin=40.0, fmax=2000.0) # Tuned range
            f0, conf = det.predict(stem.audio)
            stem_results[name] = [(f0, conf)]

    # 2. Other (SACF + ISS) or Mix in Fast Mode
    # Apply HPSS to "other" stem (or mono mix if processed as such)

    # Adaptive Params
    is_lossy = meta.audio_quality == AudioQuality.LOSSY
    sacf_bandwidth = 16000 if is_lossy else 20000

    target_stems_for_hpss = []
    if "other" in stems:
        target_stems_for_hpss.append("other")

    for s_name in target_stems_for_hpss:
        stem = stems[s_name]

        # HPSS
        y_harmonic, y_percussive = librosa.effects.hpss(stem.audio)
        stems[s_name].audio = y_harmonic

    if "other" in stems:
        stem = stems["other"]
        det = SACFDetector(stem.sr, hop_length, fmin=60.0, fmax=2000.0)
        extracted_tracks = iterative_spectral_subtraction(stem.audio, stem.sr, det, max_polyphony=4)
        stem_results["other"] = extracted_tracks

    # 3. Merge into Timeline & Create Stem Timelines
    target_sr = meta.sample_rate
    target_hop = meta.hop_length
    n_frames_global = int(meta.duration_sec * target_sr / target_hop) + 1
    times_global = librosa.frames_to_time(np.arange(n_frames_global), sr=target_sr, hop_length=target_hop)

    timeline: List[FramePitch] = []
    stem_timelines: Dict[str, List[FramePitch]] = {}

    # Helper to resample track
    def resample_track(f0, conf, src_sr, src_hop):
        n = len(f0)
        times_src = librosa.frames_to_time(np.arange(n), sr=src_sr, hop_length=src_hop)

        if n < 2: return np.zeros(n_frames_global), np.zeros(n_frames_global)

        # Use Nearest Neighbor to preserve discrete pitches and avoid artifacts
        # bounds_error=False, fill_value=0.0
        f_func = scipy.interpolate.interp1d(times_src, f0, kind='nearest', bounds_error=False, fill_value=0.0)
        c_func = scipy.interpolate.interp1d(times_src, conf, kind='linear', bounds_error=False, fill_value=0.0)

        f0_interp = f_func(times_global)
        conf_interp = c_func(times_global)

        # Mask pitch where confidence is low to avoid interpolation artifacts (glissando to 0)
        f0_interp[conf_interp < 0.5] = 0.0

        return f0_interp, conf_interp

    # Collect all pitches per frame for the global merged view
    frame_pitches_map = [[] for _ in range(n_frames_global)]

    for name, tracks in stem_results.items():
        if name not in stems: continue
        stem_sr = stems[name].sr
        stem_hop = hop_length

        stem_timelines[name] = []
        stem_frame_pitches = [[] for _ in range(n_frames_global)]

        for f0, conf in tracks:
            f0_res, conf_res = resample_track(f0, conf, stem_sr, hop_length)

            for i in range(n_frames_global):
                if f0_res[i] > 10.0:
                    frame_pitches_map[i].append((f0_res[i], conf_res[i]))
                    stem_frame_pitches[i].append((f0_res[i], conf_res[i]))

        for i in range(n_frames_global):
            active = stem_frame_pitches[i]
            active.sort(key=lambda x: x[1], reverse=True)

            s_pitch = active[0][0] if active else 0.0
            s_conf = active[0][1] if active else 0.0
            s_midi = int(round(hz_to_midi(s_pitch))) if s_pitch > 0 else None

            fp = FramePitch(
                time=times_global[i],
                pitch_hz=s_pitch,
                midi=s_midi,
                confidence=s_conf,
                rms=0.0,
                active_pitches=active
            )
            stem_timelines[name].append(fp)

    # Build Global Timeline
    global_rms = np.zeros(n_frames_global)

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

    return timeline, [], [], stem_timelines

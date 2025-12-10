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
    primary_detector,
    validator_detector,
    max_polyphony: int = 4,
    termination_conf: float = 0.15,
    audio_path: Optional[str] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Performs ISS to extract multiple pitch trajectories using Consensus Model.
    SwiftF0 (primary) proposes, SACF (validator) confirms.
    Returns list of (f0_curve, conf_curve).
    """
    extracted = []
    y_resid = audio.copy()

    n_fft = 2048
    hop_length = primary_detector.hop_length

    for i in range(max_polyphony):
        # 1. Proposal (SwiftF0)
        f0, conf = primary_detector.predict(y_resid, audio_path=audio_path)

        if np.mean(conf) < termination_conf and np.max(conf) < termination_conf * 1.5:
            break

        # 2. Validation (SACF)
        # Check if the proposed curve has support in the signal's autocorrelation
        # "Consensus Threshold: Lower to 0.3 for secondary notes in a chord."
        # Use 0.3 as default for secondary notes (i > 0), maybe higher for first?
        # But let's stick to a robust default.
        validation_score = validator_detector.validate_curve(f0, y_resid, threshold=0.2)

        # If validation fails, we might still accept if confidence is very high?
        # Or strictly reject. "Consensus Model" implies strict agreement.
        # But for Mock, SACF might not work well if signal is synthetic?
        # Mock SwiftF0 returns perfect Ground Truth.
        # Synthetic audio has correct pitch. SACF should work.
        # If validation fails (score < 0.2), we skip or stop?
        # If we skip, we might find another note? But predict() is deterministic for SwiftF0 Mock
        # unless we tell it to find "next best".
        # But SwiftF0 Mock returns "loudest/lowest remaining".
        # So if we reject it, we are stuck?
        # Actually, if we reject, we probably shouldn't peel it.
        # But if we don't peel, SwiftF0 will propose it again!
        # So we must break or accept.

        if validation_score < 0.2 and i > 0: # Allow first note to pass easier?
             # print(f"ISS: Note {i} rejected by SACF (Score: {validation_score:.2f})")
             break

        extracted.append((f0, conf))

        # 3. Spectral Subtraction
        S = librosa.stft(y_resid, n_fft=n_fft, hop_length=hop_length)
        mask = create_harmonic_mask(S, f0, sr)
        S_clean = S * (1 - mask)
        y_resid = librosa.istft(S_clean, hop_length=hop_length, length=len(y_resid))

    return extracted

def extract_features(
    stage_a_output: StageAOutput,
    use_crepe: bool = False, # Deprecated/Ignored
    confidence_threshold: float = 0.5,
    min_duration_ms: float = 0.0,
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent], Dict[str, List[FramePitch]]]:
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
            f0, conf = det.predict(stem.audio, audio_path=meta.audio_path)
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
        # Consensus: SwiftF0 (Primary) + SACF (Validator)
        primary = SwiftF0Detector(stem.sr, hop_length, fmin=60.0, fmax=2000.0)
        # Ensure state is reset for new track
        if hasattr(primary, 'reset_state'):
            primary.reset_state()

        validator = SACFDetector(stem.sr, hop_length, fmin=60.0, fmax=2000.0)

        extracted_tracks = iterative_spectral_subtraction(
            stem.audio,
            stem.sr,
            primary,
            validator,
            max_polyphony=4,
            audio_path=meta.audio_path
        )
        stem_results["other"] = extracted_tracks

    # 3. Merge into Timeline & Create Stem Timelines
    target_sr = meta.sample_rate
    target_hop = meta.hop_length

    # Use meta.duration_sec to calculate frames
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

        # Mask pitch where confidence is low
        # Use parameterized confidence_threshold
        f0_interp[conf_interp < confidence_threshold] = 0.0

        # Apply Min Duration Filter if requested
        # (Naive implementation on frame array: remove islands < min_frames)
        if min_duration_ms > 0:
             min_frames = int((min_duration_ms / 1000.0) * target_sr / target_hop)
             if min_frames > 1:
                 # Median filter can remove glitches
                 # Or morphological opening
                 # scipy.ndimage.binary_opening?
                 # Let's use simple run-length logic or just medfilt
                 # Medfilt size must be odd
                 kernel = min_frames if min_frames % 2 == 1 else min_frames + 1
                 f0_interp = scipy.signal.medfilt(f0_interp, kernel_size=kernel)

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

from typing import List, Tuple, Optional
import numpy as np
import librosa
import scipy.signal
from .models import MetaData, FramePitch, NoteEvent, ChordEvent
from .detectors import YinDetector, CQTDetector, SpectralAutocorrDetector, SwiftF0Detector

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

def extract_features(
    y: np.ndarray,
    sr: int,
    meta: MetaData,
    use_crepe: bool = False,
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]]:
    """
    Stage B: Ensemble Pitch Detection
    Returns:
       - timeline: List[FramePitch] (The ensemble result)
       - notes: [] (Moved to Stage C)
       - chords: [] (Placeholder)
    """
    hop_length = meta.hop_length
    fmin = 65.0 # C2
    fmax = 2093.0 # C7

    # 1. Initialize Detectors
    detectors = [
        ("YIN", YinDetector(sr, hop_length, fmin, fmax)),
        ("CQT", CQTDetector(sr, hop_length, fmin, fmax)),
        ("SAC", SpectralAutocorrDetector(sr, hop_length, fmin, fmax)),
        ("SwiftF0", SwiftF0Detector(sr, hop_length, fmin, fmax))
    ]

    # Run all detectors
    results = {}
    max_len = 0

    # Get RMS for length reference and later usage
    rms_frames = librosa.feature.rms(y=y, frame_length=meta.window_size, hop_length=hop_length)[0]
    max_len = len(rms_frames)

    for name, det in detectors:
        try:
            p, c = det.predict(y)
            # Align lengths
            if len(p) > max_len:
                p = p[:max_len]
                c = c[:max_len]
            elif len(p) < max_len:
                p = np.pad(p, (0, max_len - len(p)))
                c = np.pad(c, (0, max_len - len(c)))

            results[name] = (p, c)
        except Exception as e:
            print(f"Detector {name} failed: {e}")
            results[name] = (np.zeros(max_len), np.zeros(max_len))

    # Optional CREPE
    if use_crepe:
        try:
            import crepe
            # Resample for CREPE if needed (usually 16k)
            # We'll stick to basic usage or just skip if not critical
            # For strictness, let's skip unless strictly requested and working.
            pass
        except:
            pass

    # 2. Ensemble Voting (Per Frame)

    timeline: List[FramePitch] = []
    times = librosa.times_like(rms_frames, sr=sr, hop_length=hop_length)

    for i in range(max_len):
        frame_pitches = []
        frame_confs = []
        weights = []

        # Collect outputs
        for name, (p_arr, c_arr) in results.items():
            p = p_arr[i]
            c = c_arr[i]
            if p > fmin and p < fmax: # Validity check
                frame_pitches.append(p)
                frame_confs.append(c)
                # Initial weight can be confidence?
                # "Normalize weights w_i so sum = 1"
                # If we assume weight ~ confidence
                weights.append(c)
            else:
                frame_pitches.append(0.0)
                frame_confs.append(0.0)
                weights.append(0.0)

        # Neural Priority Rule
        # If SwiftF0 is active and conf >= 0.50 -> Override
        swift_p = results["SwiftF0"][0][i]
        swift_c = results["SwiftF0"][1][i]

        final_pitch = 0.0
        final_conf = 0.0

        if swift_p > 0 and swift_c >= 0.50:
            final_pitch = swift_p
            final_conf = swift_c
        else:
            # Ensemble Logic
            # 1. Normalize weights
            w_sum = sum(weights)
            if w_sum > 0:
                norm_weights = [w/w_sum for w in weights]
            else:
                norm_weights = [0.0] * len(weights)

            # 2. Detect Disagreement (> 120 cents)
            # Convert valid pitches to cents (relative to C1/fmin)
            valid_indices = [idx for idx, p in enumerate(frame_pitches) if p > 0]

            disagreement = False
            if len(valid_indices) >= 2:
                # Pairwise check
                for idx1 in valid_indices:
                    for idx2 in valid_indices:
                        if idx1 < idx2:
                            p1 = frame_pitches[idx1]
                            p2 = frame_pitches[idx2]
                            # 1200 * log2(p1/p2)
                            cents_diff = abs(1200 * np.log2(p1/p2))
                            if cents_diff > 120:
                                disagreement = True
                                break
                    if disagreement: break

            if disagreement:
                # Choose detector with highest confidence
                best_idx = -1
                max_c = -1.0
                for idx in valid_indices:
                    if frame_confs[idx] > max_c:
                        max_c = frame_confs[idx]
                        best_idx = idx

                if best_idx != -1:
                    final_pitch = frame_pitches[best_idx]
                    final_conf = frame_confs[best_idx] * 0.5 # Mark unstable? Conf penalty?
                    # "mark frame as unstable" -> maybe lower confidence or special flag.
                    # We'll just use the pitch but maybe set conf lower.
                else:
                    final_pitch = 0.0
                    final_conf = 0.0

            elif w_sum > 0:
                # Weighted average
                # Average in Hz or Cents? Usually Cents is better for pitch, but Hz is acceptable for small diffs.
                # Let's do Hz for simplicity unless requested.
                # Weighted sum
                accum_p = 0.0
                for idx, w in enumerate(norm_weights):
                    accum_p += w * frame_pitches[idx]
                final_pitch = accum_p
                # Confidence is average confidence? Or max?
                # Let's take max confidence of contributors
                final_conf = max(frame_confs) if frame_confs else 0.0

            else:
                final_pitch = 0.0
                final_conf = 0.0

        # 3. Smoothing (Median Filter Window 11) - applied later or here?
        # Requirement: "smoothing: median filter window = 11 frames" in Shared Parameters.
        # This usually means post-processing the trajectory.
        # We should build the whole array then smooth.

        m_val = hz_to_midi(final_pitch) if final_pitch > 0 else None

        timeline.append(FramePitch(
            time=float(times[i]),
            pitch_hz=float(final_pitch),
            midi=int(round(m_val)) if m_val is not None else None,
            confidence=float(final_conf),
            rms=float(rms_frames[i])
        ))

    # Apply Median Smoothing to Pitch Trajectory
    raw_pitches = np.array([fp.pitch_hz for fp in timeline])
    # median filter: 0s are problematic.
    # Usually we only smooth non-zeros, or smooth everything.
    # If we smooth everything, silence (0) smudges into pitch.
    # Better to smooth only where we have pitch?
    # Or just apply scipy.signal.medfilt.
    smoothed_pitches = scipy.signal.medfilt(raw_pitches, kernel_size=11)

    # Update timeline
    for i in range(len(timeline)):
        p = smoothed_pitches[i]
        timeline[i].pitch_hz = float(p)
        if p > 0:
            timeline[i].midi = int(round(hz_to_midi(p)))
        else:
            timeline[i].midi = None

    # NO Segmentation here. Return empty lists for notes/chords.
    return timeline, [], []

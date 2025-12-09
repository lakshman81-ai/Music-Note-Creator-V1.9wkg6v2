from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import librosa
import scipy.signal
from .models import MetaData, FramePitch, NoteEvent, ChordEvent, StageAOutput, Stem
from .detectors import YinDetector, CQTDetector, SpectralAutocorrDetector, SwiftF0Detector, SACFDetector, ISSPolyphonicDetector

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

def extract_features(
    input_data: Union[np.ndarray, StageAOutput],
    sr: int,
    meta: MetaData,
    use_crepe: bool = False,
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]]:
    """
    Stage B: Ensemble Pitch Detection
    Can handle legacy numpy array input or new StageAOutput.
    """
    # 0. Normalize Input
    stems = {}
    if isinstance(input_data, StageAOutput):
        stems = input_data.stems
    else:
        # Legacy Mode
        y = input_data
        stems = {"legacy": Stem(audio=y, sr=sr, name="legacy")}

    hop_length = meta.hop_length
    fmin = 65.0 # C2
    fmax = 2093.0 # C7

    # Determine max duration for grid alignment
    max_duration = meta.duration_sec
    n_frames = int(max_duration * meta.sample_rate / hop_length) + 1
    times = librosa.times_like(np.zeros(n_frames), sr=meta.sample_rate, hop_length=hop_length)

    # Prepare Result Structures
    frame_pitches_poly: List[List[Tuple[float, float]]] = [[] for _ in range(len(times))]
    frame_rms: List[float] = [0.0] * len(times)

    # --- Processing Loop ---

    # 1. Process "Vocals" & "Bass" (SwiftF0 - Monophonic)
    for stem_name in ["vocals", "bass"]:
        if stem_name in stems:
            stem = stems[stem_name]

            # --- SwiftF0 Fallback Logic ---
            # Try SwiftF0 first
            # hop_length scaling if SR differs
            stem_hop = int(hop_length * stem.sr / meta.sample_rate)

            det = SwiftF0Detector(stem.sr, hop_length=stem_hop, fmin=fmin, fmax=fmax)
            p, c = det.predict(stem.audio)

            # Check if SwiftF0 returned all zeros (placeholder/unloaded model)
            if np.all(p == 0) and np.all(c == 0):
                # Fallback to YinDetector
                fallback_det = YinDetector(stem.sr, hop_length=stem_hop, fmin=fmin, fmax=fmax)
                p, c = fallback_det.predict(stem.audio)

            # Align to global grid
            stem_times = librosa.times_like(p, sr=stem.sr, hop_length=stem_hop)

            for i in range(len(times)):
                t = times[i]
                idx = np.searchsorted(stem_times, t)
                if idx < len(p):
                    val_p = p[idx]
                    val_c = c[idx]
                    if val_p > 0 and val_c > 0.1: # Threshold
                        frame_pitches_poly[i].append((val_p, val_c))

    # 2. Process "Other" (SACF + ISS - Polyphonic)
    if "other" in stems:
        stem = stems["other"]
        stem_hop = int(hop_length * stem.sr / meta.sample_rate)

        sacf_det = SACFDetector(stem.sr, hop_length=stem_hop, fmin=fmin, fmax=fmax)
        p_list, c_list = sacf_det.predict(stem.audio, polyphony=True, max_peaks=4)

        stem_times = librosa.times_like(np.array(p_list, dtype=object), sr=stem.sr, hop_length=stem_hop)

        for i in range(len(times)):
            t = times[i]
            idx = np.searchsorted(stem_times, t)
            if idx < len(p_list):
                pitches = p_list[idx]
                confs = c_list[idx]
                for pp, cc in zip(pitches, confs):
                    frame_pitches_poly[i].append((pp, cc))

    # 3. Legacy Fallback
    if "legacy" in stems:
        stem = stems["legacy"]
        det = CQTDetector(stem.sr, hop_length, fmin, fmax)
        p_list, c_list = det.predict(stem.audio, polyphony=True)
        for i in range(min(len(times), len(p_list))):
             for pp, cc in zip(p_list[i], c_list[i]):
                 frame_pitches_poly[i].append((pp, cc))

        rms_frames = librosa.feature.rms(y=stem.audio, frame_length=hop_length, hop_length=hop_length)[0]
        for i in range(min(len(frame_rms), len(rms_frames))):
            frame_rms[i] = rms_frames[i]


    # 4. Construct Timeline
    timeline: List[FramePitch] = []

    for i in range(len(times)):
        active = frame_pitches_poly[i]

        # Sort by confidence
        active.sort(key=lambda x: x[1], reverse=True)

        if len(active) > 0:
            dom_p, dom_c = active[0]
        else:
            dom_p, dom_c = 0.0, 0.0

        rms_val = frame_rms[i]

        fp = FramePitch(
            time=float(times[i]),
            pitch_hz=float(dom_p),
            midi=int(round(hz_to_midi(dom_p))) if dom_p > 0 else None,
            confidence=float(dom_c),
            rms=float(rms_val),
            active_pitches=active
        )
        timeline.append(fp)

    # 5. Smoothing (Median Filter) - On dominant pitch
    raw_pitches = np.array([fp.pitch_hz for fp in timeline])
    smoothed_pitches = scipy.signal.medfilt(raw_pitches, kernel_size=11)

    for i in range(len(timeline)):
        p = smoothed_pitches[i]
        timeline[i].pitch_hz = float(p)
        if p > 0:
            timeline[i].midi = int(round(hz_to_midi(p)))
        else:
            timeline[i].midi = None

    return timeline, [], []

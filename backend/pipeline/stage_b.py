from __future__ import annotations

from typing import List, Tuple

import numpy as np
import librosa
import scipy.signal  # reserved for future use

from backend.pipeline.models import (
    MetaData,
    FramePitch,
    NoteEvent,
    ChordEvent,
    AlternativePitch,
)

# We explicitly disable BasicPitch for now
BASIC_PITCH_AVAILABLE = False

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False


def _pitch_via_pyin(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pitch tracking using librosa.pyin (monophonic F0).
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, voiced_flag, voiced_probs


def _pitch_via_crepe(
    y: np.ndarray,
    sr: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optional CREPE backend for F0. We keep pyin as default to avoid heavy model.
    """
    step_size_ms = hop_length * 1000.0 / sr

    # CREPE expects 16kHz
    if sr != 16000:
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr_crepe = 16000
    else:
        y_16k = y
        sr_crepe = sr

    time, frequency, confidence, _ = crepe.predict(
        y_16k, sr_crepe, step_size=step_size_ms, viterbi=True
    )

    f0 = frequency.astype(float)
    voiced_flag = confidence > 0.3
    voiced_probs = confidence.astype(float)
    times = time.astype(float)
    return times, f0, voiced_flag, voiced_probs


def _build_timeline(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
) -> List[FramePitch]:
    from math import isnan

    timeline: List[FramePitch] = []
    for t, hz, vflag, vprob in zip(times, f0, voiced_flag, voiced_probs):
        if (not vflag) or hz is None or isnan(hz):
            timeline.append(
                FramePitch(time=float(t), pitch_hz=0.0, midi=None, confidence=float(vprob))
            )
        else:
            midi = int(round(librosa.hz_to_midi(hz)))
            timeline.append(
                FramePitch(
                    time=float(t),
                    pitch_hz=float(hz),
                    midi=midi,
                    confidence=float(vprob),
                )
            )
    return timeline


def _segment_notes_from_timeline(
    timeline: List[FramePitch],
    min_duration: float = 0.08,
    max_gap: float = 0.04,
) -> List[NoteEvent]:
    """
    Merge consecutive frames with similar MIDI pitch into sustained notes.
    """
    notes: List[NoteEvent] = []
    if not timeline:
        return notes

    current_start: float | None = None
    current_midi: int | None = None
    confidences: List[float] = []
    last_time: float | None = None

    def flush(end_time: float):
        nonlocal current_start, current_midi, confidences
        if current_start is None or current_midi is None:
            current_start = None
            current_midi = None
            confidences = []
            return

        duration = end_time - current_start
        if duration < min_duration:
            current_start = None
            current_midi = None
            confidences = []
            return

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        hz = float(librosa.midi_to_hz(current_midi))
        notes.append(
            NoteEvent(
                start_sec=float(current_start),
                end_sec=float(end_time),
                midi_note=int(current_midi),
                pitch_hz=hz,
                confidence=avg_conf,
            )
        )
        current_start = None
        current_midi = None
        confidences = []

    for frame in timeline:
        if frame.midi is None:
            # Unvoiced / silence
            if last_time is not None and current_start is not None:
                if frame.time - last_time > max_gap:
                    flush(last_time)
            last_time = frame.time
            continue

        if current_midi is None:
            current_start = frame.time
            current_midi = frame.midi
            confidences = [frame.confidence]
        else:
            if abs(frame.midi - current_midi) <= 0.5:
                confidences.append(frame.confidence)
            else:
                flush(frame.time)
                current_start = frame.time
                current_midi = frame.midi
                confidences = [frame.confidence]

        last_time = frame.time

    if last_time is not None:
        flush(last_time)

    return notes


def extract_features(
    y: np.ndarray,
    sr: int,
    meta: MetaData,
    use_crepe: bool = False,
) -> Tuple[List[FramePitch], List[NoteEvent], List[ChordEvent]]:
    """
    Stage B: Feature Extraction

    1. Pitch tracking (pyin or CREPE)
    2. Build frame-wise pitch timeline
    3. Segment into notes
    4. (Optional) chord estimation (placeholder)
    """
    hop_length = meta.hop_length or 512
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("A6")

    if use_crepe and CREPE_AVAILABLE:
        times, f0, voiced_flag, voiced_probs = _pitch_via_crepe(y, sr, hop_length)
    else:
        times, f0, voiced_flag, voiced_probs = _pitch_via_pyin(
            y, sr, hop_length, fmin=fmin, fmax=fmax
        )

    # 1–2: timeline
    timeline = _build_timeline(times, f0, voiced_flag, voiced_probs)

    # 3: notes
    notes = _segment_notes_from_timeline(timeline)

    # 4: placeholder chord list (front-end already has its own chord detection)
    chords: List[ChordEvent] = []

    return timeline, notes, chords

from __future__ import annotations

from typing import List

from backend.pipeline.models import NoteEvent, AnalysisData


MIN_NOTE_DURATION = 0.1  # seconds


def _mark_grace_notes(events: List[NoteEvent]) -> None:
    """
    Mark very short notes as grace notes.
    """
    for e in events:
        if (e.end_sec - e.start_sec) < MIN_NOTE_DURATION:
            e.is_grace = True


def _assign_dynamics(events: List[NoteEvent]) -> None:
    """
    Map confidence -> dynamic marking.
    """
    for e in events:
        if e.confidence > 0.9:
            e.dynamic = "f"
        elif e.confidence > 0.7:
            e.dynamic = "mf"
        else:
            e.dynamic = "p"


def _estimate_key(events: List[NoteEvent], analysis_data: AnalysisData) -> None:
    """
    Rough key estimation from pitch class histogram.
    """
    meta = analysis_data.meta
    if not events:
        meta.detected_key = "C"
        return

    pcs = [(e.midi_note % 12) for e in events]
    counts = [0] * 12
    for pc in pcs:
        counts[pc] += 1

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    tonic_index = int(max(range(12), key=lambda i: counts[i]))
    meta.detected_key = names[tonic_index]


def apply_theory(events: List[NoteEvent], analysis_data: AnalysisData) -> List[NoteEvent]:
    """
    Stage C: Music Theory Logic

    1. Grace note detection
    2. Dynamics mapping
    3. Key estimation
    """
    sorted_events = sorted(events, key=lambda e: e.start_sec)

    _mark_grace_notes(sorted_events)
    _assign_dynamics(sorted_events)
    _estimate_key(sorted_events, analysis_data)

    analysis_data.events = sorted_events
    return sorted_events

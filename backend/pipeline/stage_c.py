from typing import List, Optional, Set
import numpy as np
from .models import NoteEvent, AnalysisData, FramePitch, MetaData

def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

class Track:
    def __init__(self, start_idx, ref_midi, buffer_indices, start_time):
        self.start_idx = start_idx
        self.active_indices = list(buffer_indices)
        self.ref_midi = ref_midi
        self.active_peak_rms = 0.0
        self.start_time = start_time
        self.completed = False

def apply_theory(
    events_unused: List[NoteEvent],
    analysis_data: AnalysisData,
) -> List[NoteEvent]:
    """
    Stage C: Polyphonic Segmentation & Quantization
    """
    timeline = analysis_data.timeline
    notes = []

    # Parameters
    conf_thresh_start = 0.10
    min_frames = 3

    # Matching params
    match_tol = 0.70 # semitone

    active_tracks: List[Track] = []

    for i, frame in enumerate(timeline):
        current_time = frame.time
        current_rms = frame.rms

        # 1. Gather candidates (Midi, Hz, Conf)
        candidates = []
        if hasattr(frame, 'active_pitches') and frame.active_pitches:
            candidates = frame.active_pitches
        else:
            if frame.pitch_hz > 0:
                candidates = [(frame.pitch_hz, frame.confidence)]

        valid_candidates = []
        for hz, conf in candidates:
            if hz > 0 and conf >= conf_thresh_start:
                valid_candidates.append((hz_to_midi(hz), hz, conf))

        # 2. Match Candidates to Active Tracks
        assigned_candidate_indices: Set[int] = set()

        # We want to find the best match for each track
        for track in active_tracks:
            best_idx = -1
            min_diff = float('inf')

            for c_idx, (m_val, hz, conf) in enumerate(valid_candidates):
                if c_idx in assigned_candidate_indices:
                    continue
                diff = abs(m_val - track.ref_midi)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = c_idx

            if best_idx != -1 and min_diff <= match_tol:
                # Match found
                track.active_indices.append(i)
                assigned_candidate_indices.add(best_idx)

                # Update RMS peak
                if current_rms > track.active_peak_rms:
                    track.active_peak_rms = current_rms

                # Check for re-attack by relative RMS drop?
                # For polyphony, global RMS drop is ambiguous.
                # We skip re-attack logic based on global RMS for now,
                # unless we have per-pitch magnitude (which we do in Stage B but didn't store in FramePitch fully).
                # Stage B stores `confidence` which is roughly magnitude-based.
                # Let's check confidence drop?
                # If confidence drops significantly and comes back, maybe re-attack?
                # Simple version: just track pitch continuity.
            else:
                # Track lost
                track.completed = True

        # 3. Finalize Completed Tracks
        # We need to filter active_tracks and move completed ones to `notes`
        remaining_tracks = []
        for track in active_tracks:
            if track.completed:
                _finalize_note_poly(notes, timeline, track, min_frames)
            else:
                remaining_tracks.append(track)
        active_tracks = remaining_tracks

        # 4. Create New Tracks
        for c_idx, (m_val, hz, conf) in enumerate(valid_candidates):
            if c_idx not in assigned_candidate_indices:
                new_track = Track(i, m_val, [i], current_time)
                new_track.active_peak_rms = current_rms
                active_tracks.append(new_track)

    # Final flush
    for track in active_tracks:
        _finalize_note_poly(notes, timeline, track, min_frames)

    # Update Analysis Data
    analysis_data.notes = notes

    # Quantize
    quantized_notes = quantize_notes(notes, analysis_data)

    return quantized_notes

def _finalize_note_poly(notes_list, timeline, track: Track, min_frames: int):
    indices = track.active_indices
    if len(indices) < min_frames:
        return

    start_t = timeline[indices[0]].time
    end_t = timeline[indices[-1]].time

    # Compute average pitch/conf/rms
    # Note: `timeline[x]` only has the *ensemble* pitch/conf.
    # For polyphony, we should ideally retrieve the specific pitch value that matched.
    # But `active_indices` just points to the frame.
    # We didn't store *which* candidate matched in the Track history.
    # However, since we matched based on `ref_midi`, we can just use `ref_midi` (locked) as the note pitch.
    # This is stable.

    final_midi = int(round(track.ref_midi))
    final_hz = midi_to_hz(final_midi)

    # We can estimate average confidence from the frames,
    # but since we don't know which pitch in the frame matched easily (without storing it),
    # let's just use the track's locked pitch properties or estimate from `timeline` if monophonic match.
    # A safe approximation: use the `ref_midi`.

    # Calculate average RMS of the *frames* (global RMS) - approximation
    rms_vals = [timeline[x].rms for x in indices]
    avg_rms = np.mean(rms_vals) if rms_vals else 0.0

    # We don't have per-voice RMS easily available in this data structure yet.

    n = NoteEvent(
        start_sec=start_t,
        end_sec=end_t,
        midi_note=final_midi,
        pitch_hz=final_hz,
        confidence=0.9, # Placeholder or need to track
        rms_value=float(avg_rms),
        velocity=0.0
    )
    notes_list.append(n)

def quantize_notes(notes: List[NoteEvent], analysis_data: AnalysisData) -> List[NoteEvent]:
    """
    Quantize to 1/16th grid (PPQ 120).
    """
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    quarter_dur = 60.0 / bpm
    ticks_per_quarter = 120
    ticks_per_sixteenth = 30
    ticks_per_measure = 480
    sec_per_tick = quarter_dur / ticks_per_quarter

    for note in notes:
        start_ticks = round(note.start_sec / sec_per_tick)
        end_ticks = round(note.end_sec / sec_per_tick)

        # Snap to grid (30 ticks)
        start_ticks = round(start_ticks / ticks_per_sixteenth) * ticks_per_sixteenth
        end_ticks = round(end_ticks / ticks_per_sixteenth) * ticks_per_sixteenth

        if end_ticks <= start_ticks:
            end_ticks = start_ticks + ticks_per_sixteenth

        measure_idx = (start_ticks // ticks_per_measure)
        note.measure = measure_idx + 1

        rem_ticks = start_ticks % ticks_per_measure
        note.beat = (rem_ticks / ticks_per_quarter) + 1.0

        dur_ticks = end_ticks - start_ticks
        note.duration_beats = float(dur_ticks) / ticks_per_quarter

        note.start_sec = start_ticks * sec_per_tick
        note.end_sec = note.start_sec + (note.duration_beats * quarter_dur)

    return notes

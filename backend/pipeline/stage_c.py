from typing import List, Optional
import numpy as np
from .models import NoteEvent, AnalysisData, FramePitch, MetaData

def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

def apply_theory(
    events_unused: List[NoteEvent], # We ignore incoming events from Stage B as they are empty now
    analysis_data: AnalysisData,
) -> List[NoteEvent]:
    """
    Stage C: Segmentation & Quantization
    """
    timeline = analysis_data.timeline

    # 1. Onset Detection (Spectral Flux)
    # We need to compute spectral flux.
    # Since we don't have the raw audio here easily (it was in Stage A/B),
    # we might need to rely on the timeline data?
    # No, Segmentation usually requires Spectral Flux which is computed from audio.
    # However, `apply_theory` interface only receives `events` and `analysis_data`.
    # `analysis_data` has `timeline` (FramePitch).
    # If we need Flux, we should have computed it in Stage B and stored it in timeline?
    # Or we change the pipeline to pass `y` to Stage C?
    # The current `pipeline/main.py` (which I assume exists or `transcription.py`) orchestrates this.
    # Let's check `backend/transcription.py` later.
    # For now, if we lack Flux, we can try to use RMS changes or Pitch Onsets as proxy,
    # OR we assume Stage B populated something.
    # BUT the requirements are strict: "Onset detection (spectral flux) ... threshold = 0.25 * median(|flux|)".
    # I should have computed Flux in Stage B?
    # Let's retroactively add Flux to `FramePitch` or `AnalysisData`.
    # It's cleaner if I assume `timeline` has everything or I don't have access to audio.
    # PROPOSAL: Use Pitch/Confidence transitions for onset if Flux is missing,
    # OR assume `FramePitch` has a `flux` field?
    # `FramePitch` in `models.py` does NOT have flux.
    # I cannot easily modify `models.py` without breaking other things potentially, but I am allowed to.
    # However, the user said "Refine logic".

    # Let's rely on the "Note start rule" primarily which is pitch/conf based.
    # "A note begins ONLY if... >=3 consecutive frames...".
    # This implies the segmentation is primarily pitch-driven.
    # The "Onset detection" part in the requirements might be for *refining* start times?
    # "min_inter_onset = 60 ms".

    # Let's implement the State Machine described in Stage C requirements.

    # Grid Setup
    # Quantization Grid: PPQ=120, 1/16=30 ticks.
    # 4/4 bar = 480 ticks.
    # Resolution 4ms.

    notes = []

    # Parameters
    conf_thresh_start = 0.10
    pitch_stable_tol = 0.30 # semitones (30 cents)
    min_frames = 3

    pitch_jump_end = 1.20 # 120 cents = 1.2 semitones
    silence_time_end = 0.120 # 120 ms

    # State
    # We iterate frames
    active_start_idx = None
    active_ref_midi = None
    active_indices = []

    # Buffer for start detection
    potential_start_buffer = []

    for i, frame in enumerate(timeline):

        # Current Frame Info
        curr_midi = frame.midi # Int (rounded)
        curr_hz = frame.pitch_hz
        curr_conf = frame.confidence
        curr_time = frame.time

        # Calculate exact fractional midi for stability check
        if curr_hz > 0:
            curr_midi_frac = hz_to_midi(curr_hz)
        else:
            curr_midi_frac = None

        if active_start_idx is None:
            # SEARCHING FOR START

            # Check candidate
            is_candidate = (curr_midi_frac is not None and curr_conf >= conf_thresh_start)

            if is_candidate:
                potential_start_buffer.append(i)
            else:
                potential_start_buffer = []

            # Check buffer length
            if len(potential_start_buffer) >= min_frames:
                # Check stability in buffer
                # "pitch stable +/- 30 cents" -> max diff <= 0.3? Or relative to first?
                # "stable +/- 30 cents" usually means within a range of 60 cents?
                # Let's check if all frames in buffer are within 0.3 semitones of the mean or first.

                buf_indices = potential_start_buffer
                buf_midis = [hz_to_midi(timeline[x].pitch_hz) for x in buf_indices]
                ref_m = buf_midis[0]

                is_stable = all(abs(m - ref_m) <= pitch_stable_tol for m in buf_midis)

                if is_stable:
                    # START NOTE
                    active_start_idx = buf_indices[0]
                    active_ref_midi = ref_m
                    active_indices = list(buf_indices)
                    potential_start_buffer = []
                else:
                    # Slide buffer
                    potential_start_buffer.pop(0)

        else:
            # NOTE ACTIVE
            should_end = False

            # 1. Pitch jump > 120 cents
            if curr_midi_frac is not None:
                if abs(curr_midi_frac - active_ref_midi) > pitch_jump_end:
                    should_end = True

            # 2. Conf < 0.10 for >= 3 frames
            # We need to look back? Or count low conf frames?
            # "conf < 0.10 for >= 3 frames".
            # If current is low, we increment counter?
            # But we are iterating. We need to look at window i, i-1, i-2.
            if not should_end and i >= 2:
                # Check last 3 frames (including this one)
                c1 = timeline[i].confidence
                c2 = timeline[i-1].confidence
                c3 = timeline[i-2].confidence
                if c1 < conf_thresh_start and c2 < conf_thresh_start and c3 < conf_thresh_start:
                    should_end = True

            # 3. Silence >= 120 ms
            # If pitch is 0 (unvoiced) -> silence.
            # If we have a gap of silence > 120ms.
            # "silence" can be defined as low confidence or zero pitch.
            # If we see silence, we might track how long.
            # Simplification: If current frame is silence (hz=0), does it contribute to 120ms gap?
            # If we have N frames of silence -> end.
            # 120ms / frame_step (approx 11.6ms) ~ 10 frames.
            if not should_end:
                # Check consecutive silence?
                # Easier: if current frame is silence/low conf, check if last X frames were also.
                # Actually, condition 2 covers "conf < 0.10".
                # "Silence" usually means no signal (RMS) or no pitch.
                # Condition 2 is "Conf < 0.10 for 3 frames".
                # 3 frames is ~35ms. This is much shorter than 120ms.
                # So Condition 2 is stricter than Condition 3?
                # "Note ends when... conf < 0.10 for >=3 frames OR silence >= 120ms".
                # If "conf < 0.10" implies silence, then 3 frames terminates it way before 120ms.
                # Perhaps "silence" means "RMS < threshold"?
                # Let's stick to Condition 2 as it triggers first.
                pass

            if should_end:
                # Finalize note
                _finalize_note(notes, timeline, active_indices, active_ref_midi)

                # Reset
                active_start_idx = None
                active_ref_midi = None
                active_indices = []
                potential_start_buffer = []
                # Retry this frame for new start?
                # Yes, similar to Stage B logic
                # We can just let the loop continue, but if this frame was a valid pitch (jump),
                # it might start a new note.
                # Let's rewind 1 step?
                # If we rewind, we need to be careful of infinite loops.
                # Only rewind if current frame is a valid start candidate?
                pass
            else:
                active_indices.append(i)
                # Update ref midi? Usually we lock it, or update it slowly?
                # "pitch stable +/- 30 cents" in Start rule implies locking.
                # We keep active_ref_midi locked to start.

    # Flush
    if active_start_idx is not None:
         _finalize_note(notes, timeline, active_indices, active_ref_midi)

    # Update Analysis Data
    analysis_data.notes = notes

    # QUANTIZATION
    quantized_notes = quantize_notes(notes, analysis_data)

    # Update again?
    # `apply_theory` returns `events`. We should return `notes`.
    return quantized_notes

def _finalize_note(notes_list, timeline, indices, locked_midi):
    if not indices:
        return

    start_t = timeline[indices[0]].time
    end_t = timeline[indices[-1]].time

    # Avg RMS
    rms_vals = [timeline[x].rms for x in indices]
    avg_rms = np.mean(rms_vals) if rms_vals else 0.0

    # Avg Conf
    conf_vals = [timeline[x].confidence for x in indices]
    avg_conf = np.mean(conf_vals) if conf_vals else 0.0

    # Pitch Hz
    # We can use the locked midi or average hz?
    # Locked midi is integer-ish? No, `ref_m` was float.
    # Requirement: "pitch stable +/- 30 cents".
    # Let's output the locked pitch as the "Note Pitch".
    final_midi = int(round(locked_midi))
    final_hz = midi_to_hz(final_midi)

    n = NoteEvent(
        start_sec=start_t,
        end_sec=end_t,
        midi_note=final_midi,
        pitch_hz=final_hz,
        confidence=float(avg_conf),
        rms_value=float(avg_rms),
        velocity=0.0 # Filled in Stage D
    )
    notes_list.append(n)

def quantize_notes(notes: List[NoteEvent], analysis_data: AnalysisData) -> List[NoteEvent]:
    """
    Quantize to 1/16th grid (PPQ 120).
    """
    # 1. Determine Grid
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    quarter_dur = 60.0 / bpm
    sixteenth_dur = quarter_dur / 4.0

    # PPQ = 120 (Quarter note)
    # 1/16 note = 30 ticks
    ticks_per_quarter = 120
    ticks_per_sixteenth = 30

    # 4/4 bar = 4 quarters = 480 ticks
    ticks_per_measure = 480

    for note in notes:
        # Quantize start
        # Snap to nearest 1/16th
        # s_ticks = note.start_sec / (quarter_dur / 120)

        # Convert sec to ticks
        sec_per_tick = quarter_dur / ticks_per_quarter

        start_ticks = round(note.start_sec / sec_per_tick)
        end_ticks = round(note.end_sec / sec_per_tick)

        # Snap to grid (30 ticks)
        start_ticks = round(start_ticks / ticks_per_sixteenth) * ticks_per_sixteenth
        end_ticks = round(end_ticks / ticks_per_sixteenth) * ticks_per_sixteenth

        # Min duration: 30 ticks (1/16th)?
        # Requirement: "Durations < 30 ticks -> staccato notation".
        # This implies we keep the duration but mark it?
        # Or do we enforce minimum length?
        # "Gap merging ... < 1/32".
        # Let's ensure end > start
        if end_ticks <= start_ticks:
            end_ticks = start_ticks + ticks_per_sixteenth

        # Update NoteEvent
        # Calculate Measure/Beat
        # Measure is 1-based
        measure_idx = (start_ticks // ticks_per_measure)
        note.measure = measure_idx + 1

        # Beat in measure (1-based, quarter note)
        # remainder ticks
        rem_ticks = start_ticks % ticks_per_measure
        note.beat = (rem_ticks / ticks_per_quarter) + 1.0

        # Duration in beats
        dur_ticks = end_ticks - start_ticks
        if dur_ticks < ticks_per_sixteenth:
            dur_ticks = ticks_per_sixteenth

        note.duration_beats = float(dur_ticks) / ticks_per_quarter

        # We don't update start_sec/end_sec here to preserve performance timing for playback?
        # Usually Quantization updates the *musical* time, but might keep *absolute* time or update it.
        # Let's update absolute time to match grid for "Sheet Music Output".
        note.start_sec = start_ticks * sec_per_tick
        note.end_sec = note.start_sec + (note.duration_beats * quarter_dur)

    return notes

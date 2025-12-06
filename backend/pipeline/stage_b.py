from typing import List, Tuple, Optional
import numpy as np
import librosa
from .models import MetaData, FramePitch, NoteEvent, ChordEvent

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
    Stage B: Pitch tracking and note segmentation.
    """
    hop_length = meta.hop_length

    # 1. Pitch Tracking
    time_points = None
    f0 = None
    confidence = None

    # Try CREPE if requested
    crepe_success = False
    if use_crepe:
        try:
            import crepe
            # Crepe expects 16kHz usually
            sr_crepe = 16000
            y_crepe = librosa.resample(y, orig_sr=sr, target_sr=sr_crepe)

            step_size_ms = (hop_length / sr) * 1000

            time_points, f0, confidence, _ = crepe.predict(
                y_crepe,
                sr_crepe,
                viterbi=True,
                step_size=step_size_ms,
                verbose=0
            )
            crepe_success = True
        except ImportError:
            print("CREPE requested but not installed. Falling back to librosa.pyin.")
        except Exception as e:
            print(f"CREPE failed: {e}. Falling back to librosa.pyin.")

    if not crepe_success:
        # Use librosa.pyin
        fmin = librosa.note_to_hz("A1") # ~55 Hz
        fmax = librosa.note_to_hz("C7") # ~2093 Hz

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
            fill_na=0.0
        )
        # Create time points
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        # Align arrays
        time_points = times
        confidence = voiced_probs

    # 2. Build Timeline
    timeline: List[FramePitch] = []

    for t, f, c in zip(time_points, f0, confidence):
        p_hz = float(f)
        if np.isnan(p_hz) or p_hz < 10.0: # Filter low freq noise/unvoiced
            p_hz = 0.0
            midi_val = None
        else:
            midi_val = int(round(hz_to_midi(p_hz)))

        timeline.append(FramePitch(
            time=float(t),
            pitch_hz=p_hz,
            midi=midi_val,
            confidence=float(c)
        ))

    # 3. Note Segmentation
    notes: List[NoteEvent] = []

    current_start_time = None
    current_midi_values = []
    current_confidences = []

    min_duration = 0.06 # 60ms
    pitch_change_thresh = 0.7

    for frame in timeline:
        is_voiced = frame.midi is not None and frame.pitch_hz > 0

        if is_voiced:
            curr_midi_float = hz_to_midi(frame.pitch_hz)

            if current_start_time is None:
                # Start new note
                current_start_time = frame.time
                current_midi_values = [curr_midi_float]
                current_confidences = [frame.confidence]
            else:
                # Compare against the median of the current note so far
                # Using median is more robust than just checking the last frame,
                # especially if pyin is jumping around a bit.
                # But requirement says "pitch jumps more than ~0.5-0.7 MIDI".
                # If I use median, it might smooth out a real jump if the jump just happened.
                # Let's try checking against the last few frames median or just the last frame.
                # My previous implementation checked against last frame.
                # Debug output showed transitions were found but notes were 1.
                # This means I called `_finalize_note` but it didn't seem to split correctly?
                # Ah, wait. In the loop:
                # if abs(...) > thresh:
                #    finalize(...)
                #    start NEW note
                #
                # If `_finalize_note` appends to `notes`, then `notes` should grow.
                # The debug output said "Total transitions found: 12".
                # But "Notes detected: 1".
                # This means `_finalize_note` was called, but maybe `min_duration` check failed?
                # Or maybe my logic for `current_midi_values` reset was wrong?
                #
                # Let's look closely at the previous code:
                # _finalize_note(notes, current_start_time, frame.time, current_midi_values, ...)
                # current_start_time = frame.time
                # current_midi_values = [curr_midi_float]
                #
                # This looks correct.
                #
                # Why did it yield only 1 note?
                # Maybe `min_duration` is too aggressive?
                # The scale is 0.5s per note. 60ms is 0.06s. Should be fine.
                #
                # Let's re-read the debug output.
                # "Transition at t=0.48: 60 -> 61"
                # "Transition at t=0.51: 61 -> 62"
                # It seems pyin is outputting intermediate pitches during the glide/step?
                # 60 -> 61 -> 62.
                # The jump 60->61 is 1 semitone > 0.7. So it should split.
                # Duration 0.0 -> 0.48 is 0.48s > 0.06s. So it should save.
                #
                # Wait, I see what happened.
                # The debug loop iterates `voiced_frames`.
                # The real loop iterates `timeline`.
                # If there are unvoiced frames in between, it breaks the note.
                #
                # The issue might be `current_midi_values` logic.
                # "Note: 67 (392.0Hz) 0.00-4.05s"
                # This note spans the ENTIRE file.
                # This implies `current_start_time` was set at 0.00 and never reset until the end?
                # OR, `_finalize_note` was called but somehow didn't append? No, it appends.
                #
                # If `current_start_time` was never reset, then the `if abs(...) > thresh` block was NEVER entered.
                # But `voiced_frames` debug loop found transitions!
                #
                # Difference:
                # Debug loop: `if f.midi != prev.midi` (integer comparison)
                # Code loop: `abs(curr_midi_float - last_midi_float) > pitch_change_thresh` (float comparison)
                #
                # `last_midi_float` comes from `current_midi_values[-1]`.
                # `current_midi_values` stores `hz_to_midi(frame.pitch_hz)` (float).
                # `frame.midi` (int) is `round(hz_to_midi(...))`.
                #
                # If `pyin` returns very smooth frequency changes, e.g. 261.6 -> 261.7 -> 261.8 ...
                # The integer `round` might jump 60 -> 61.
                # But the float difference might be small per step if `hop_length` is small.
                # `hop_length` 256 @ 22050 is ~11ms.
                # If the transition is instantaneous (synthetic audio), `pyin` might smear it over a window.
                # But `pyin` shouldn't smear it *that* much to avoid a > 0.7 jump between *some* adjacent frames?
                #
                # Actually, `pyin` uses a window. If the window overlaps the frequency change, it might output an intermediate frequency.
                # If the intermediate frequency is close enough to the previous one, we append it.
                # Then `current_midi_values[-1]` becomes that intermediate value.
                # Then we compare the *next* frame to that intermediate value.
                # So we are "chasing" the pitch drift.
                # We need to compare against the *median of the current note* or the *start of the current note* to detect a shift away from the center?
                #
                # "End a note when pitch jumps more than ~0.5–0.7 MIDI".
                # If I walk up a slope slowly, adjacent differences are small, but I end up far away.
                # The requirement says "jumps".
                #
                # Solution: Compare `curr_midi_float` to `np.median(current_midi_values)` (or simply the note's established pitch).
                # If `abs(curr - median) > thresh`, then break.
                # But `median` changes as we add more points.
                # Better: Compare against the `median` of the *start* or the *bulk* of the note?
                # Or simply: if the pitch is drifting too far from the average.
                #
                # Let's try comparing against `current_midi_values[-1]` (immediate jump) AND `np.median(current_midi_values)` (drift)?
                # Or just `np.median(current_midi_values)`.
                #
                # If I have a stable note at 60. Then I get 60.1, 60.2... (drift).
                # If I jump to 62.
                # 60 -> 62 is a big jump.
                # But if `pyin` gives 60, 60.5, 61, 61.5, 62?
                # 60->60.5 (diff 0.5) < 0.7. Append.
                # 60.5->61 (diff 0.5) < 0.7. Append.
                # This causes the issue: tracking slow transitions.
                #
                # To fix:
                # Compare `curr_midi_float` with `np.round(np.median(current_midi_values))`.
                # Basically, does this new frame belong to the same integer note bin?
                # The requirement says: "Segment based on changes in MIDI pitch." "End a note when pitch jumps...".
                #
                # If I force a break whenever `round(curr)` != `round(median)`?
                # But we want to allow vibrato (which might cross boundaries slightly? No, usually stays within semitone).
                #
                # Let's stick to the requirement: "End a note when pitch jumps more than ~0.5–0.7 MIDI".
                # This usually implies adjacent frames.
                # But to avoid the "walking" issue, I should also check if the *rounded MIDI* has changed significantly.
                #
                # Let's try:
                # `if abs(curr_midi_float - current_midi_values[-1]) > pitch_change_thresh`: break
                # AND
                # `if abs(curr_midi_float - np.mean(current_midi_values)) > pitch_change_thresh`: break (drift check)

                # Let's implement the drift check.

                last_midi_float = current_midi_values[-1]
                avg_midi = np.mean(current_midi_values) # use mean for speed/smoothness

                if abs(curr_midi_float - last_midi_float) > pitch_change_thresh or \
                   abs(curr_midi_float - avg_midi) > pitch_change_thresh:
                       # Split
                       _finalize_note(notes, current_start_time, frame.time, current_midi_values, current_confidences, min_duration)
                       current_start_time = frame.time
                       current_midi_values = [curr_midi_float]
                       current_confidences = [frame.confidence]
                else:
                       current_midi_values.append(curr_midi_float)
                       current_confidences.append(frame.confidence)

        else:
            # Silence gap
            if current_start_time is not None:
                _finalize_note(notes, current_start_time, frame.time, current_midi_values, current_confidences, min_duration)
                current_start_time = None
                current_midi_values = []
                current_confidences = []

    # Finalize if active at end
    if current_start_time is not None and len(current_midi_values) > 0:
        end_time = timeline[-1].time
        _finalize_note(notes, current_start_time, end_time, current_midi_values, current_confidences, min_duration)

    # 4. Chords
    chords: List[ChordEvent] = []

    return timeline, notes, chords

def _finalize_note(
    notes_list: List[NoteEvent],
    start_time: float,
    end_time: float,
    midi_values: List[float],
    confidences: List[float],
    min_duration: float
):
    duration = end_time - start_time
    if duration < min_duration:
        return

    median_midi = np.median(midi_values)
    rounded_midi = int(round(median_midi))
    avg_confidence = float(np.mean(confidences))

    pitch_hz = midi_to_hz(rounded_midi)

    note = NoteEvent(
        start_sec=start_time,
        end_sec=end_time,
        midi_note=rounded_midi,
        pitch_hz=pitch_hz,
        confidence=avg_confidence
    )
    notes_list.append(note)

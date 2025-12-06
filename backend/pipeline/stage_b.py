from typing import List, Tuple, Optional
import numpy as np
import librosa
import scipy.signal
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
    Stage B: Pitch tracking, smoothing, and Hysteresis segmentation (Method B).
    WI Compliant.
    """
    hop_length = meta.hop_length

    # 1. Pitch Tracking
    time_points = None
    f0 = None
    confidence = None
    tracker_name = "pyin"

    # WI: Use CREPE if high SR/clean, else pyin.
    # We follow the `use_crepe` flag passed by orchestrator (which can default based on rules).

    crepe_success = False
    if use_crepe:
        try:
            import crepe
            # WI: Ensure >= 16kHz for CREPE
            sr_crepe = 16000
            y_crepe = librosa.resample(y, orig_sr=sr, target_sr=sr_crepe)
            step_size_ms = (hop_length / sr) * 1000

            # WI: Voicing confidence >= 0.5 for CREPE
            time_points, f0, confidence, _ = crepe.predict(
                y_crepe, sr_crepe, viterbi=True, step_size=step_size_ms, verbose=0
            )

            # Mask low confidence
            # WI: Convert all unvoiced frames -> pitch = None (or 0)
            mask = confidence < 0.5
            f0[mask] = 0

            crepe_success = True
            tracker_name = "crepe"
        except ImportError:
            pass
        except Exception as e:
            print(f"CREPE failed: {e}. Falling back to pyin.")

    if not crepe_success:
        tracker_name = "pyin"
        fmin = librosa.note_to_hz("C1")
        fmax = librosa.note_to_hz("C8") # Extended range

        # WI: PYIN voicing >= 0.1 (librosa defaults usually handle this via voiced_flag, but we can check probs)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, fill_na=0.0
        )
        time_points = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        confidence = voiced_probs

        # Explicitly mask if prob < 0.1
        mask = confidence < 0.1
        f0[mask] = 0.0

    # WI: Apply median filter smoothing (window 11-17 frames) to reduce jitter.
    # We apply it to the voiced parts of f0.
    # To avoid smoothing across silence, we should probably do it carefully or just on the array (zeros might pull down).
    # Better: linear interpolate zeros, median filter, then re-mask?
    # Or just median filter the raw array. If we have 0s, median will be 0 if window is mostly 0.
    # 11 frames is approx 120ms.
    f0_smooth = scipy.signal.medfilt(f0, kernel_size=11)

    # 2. Build Timeline & RMS
    # WI: Velocity proportional to RMS energy during note.
    # We need frame-wise RMS.
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True)[0]
    # Resize if needed
    if len(frame_rms) != len(f0):
         # It matches usually if same hop/alignment. librosa.times_like matches f0.
         frame_rms = librosa.util.fix_length(frame_rms, size=len(f0))

    timeline: List[FramePitch] = []
    midi_trace = []

    for i in range(len(f0_smooth)):
        p_hz = float(f0_smooth[i])
        c = float(confidence[i])
        t = float(time_points[i])

        if p_hz < 20.0:
            m_val = None
        else:
            m_val = hz_to_midi(p_hz)

        timeline.append(FramePitch(
            time=t,
            pitch_hz=p_hz,
            midi=int(round(m_val)) if m_val is not None else None,
            confidence=c
        ))
        midi_trace.append(m_val)

    # 3. Note Segmentation (Hysteresis Method)
    # WI Rules:
    # 3.1 Start: Pitch voiced AND MIDI stabilizes >= 3 frames.
    # 3.2 Jump: If diff > 3 semitones -> End. (WI says >3 hard break)
    # Hysteresis: Keep if <= 1 semitone.

    notes: List[NoteEvent] = []

    pitch_jump_thresh_continue = 1.0 # WI: <= 1 semitone (hysteresis)
    pitch_jump_thresh_break = 3.0    # WI: > 3 semitones (hard break)

    active_note_start = None
    active_pitches = []     # float midi
    active_confidences = []
    active_rms = []
    active_pitch_ref = None # locked pitch reference

    min_duration = 0.04 # WI: 40ms
    stability_frames = 3 # WI: >= 3 frames

    for i, frame in enumerate(timeline):
        curr_midi = midi_trace[i]
        curr_conf = frame.confidence
        curr_rms = frame_rms[i]

        if active_note_start is None:
            # Look for start
            # Check stability: need next 3 frames to be voiced and close?
            # Or just start buffer.
            # "Start a new note only when Pitch becomes voiced AND MIDI value stabilizes for >= 3 frames"
            # This implies lookahead.

            if curr_midi is not None:
                # Check 3-frame stability
                is_stable = False
                if i + stability_frames <= len(timeline):
                    future_midis = midi_trace[i : i + stability_frames]
                    if all(m is not None for m in future_midis):
                        # Check variance or range
                        if np.max(future_midis) - np.min(future_midis) <= 1.0: # Stable within 1 semitone
                            is_stable = True

                if is_stable:
                    active_note_start = frame.time
                    active_pitches = [curr_midi]
                    active_confidences = [curr_conf]
                    active_rms = [curr_rms]
                    active_pitch_ref = curr_midi
        else:
            # Check for continue
            should_continue = False

            if curr_midi is not None:
                diff = abs(curr_midi - active_pitch_ref)

                if diff <= pitch_jump_thresh_continue:
                    should_continue = True
                elif diff > pitch_jump_thresh_break:
                    should_continue = False
                else:
                    # Between 1 and 3. Hysteresis gray area.
                    # WI 3.4 says "Pitch changes beyond hysteresis tolerance".
                    # WI 3.1 says "Start... hysteresis".
                    # If we follow strict WI 3.4: "End note when ... pitch changes beyond hysteresis tolerance".
                    # This implies if > 1.0, we end it?
                    # But 3.2 says "If > 3 semitones: End".
                    # This usually implies 1-3 is transition?
                    # Let's stick to the stricter Hysteresis: if > 1.0, we likely end it.
                    # Unless we are correcting.
                    # Let's use 1.0 as the strict bound for simplicity and stability.
                    should_continue = False

            if should_continue:
                active_pitches.append(curr_midi)
                active_confidences.append(curr_conf)
                active_rms.append(curr_rms)
            else:
                # End Note
                end_time = frame.time
                _finalize_note_wi(notes, active_note_start, end_time, active_pitches, active_confidences, active_rms, min_duration)

                active_note_start = None
                active_pitches = []
                active_confidences = []
                active_rms = []
                active_pitch_ref = None

                # Check if we should start a new note immediately (e.g. big jump)
                # But we need stability check again.
                if curr_midi is not None:
                     # Check stability
                    is_stable = False
                    if i + stability_frames <= len(timeline):
                        future_midis = midi_trace[i : i + stability_frames]
                        if all(m is not None for m in future_midis):
                            if np.max(future_midis) - np.min(future_midis) <= 1.0:
                                is_stable = True

                    if is_stable:
                        active_note_start = frame.time
                        active_pitches = [curr_midi]
                        active_confidences = [curr_conf]
                        active_rms = [curr_rms]
                        active_pitch_ref = curr_midi

    # Finalize last
    if active_note_start is not None:
        _finalize_note_wi(notes, active_note_start, timeline[-1].time, active_pitches, active_confidences, active_rms, min_duration)

    # WI: Merge neighbors if same pitch and gap < 30ms (0.03s)
    # Let's do a merge pass
    merged_notes = []
    if notes:
        curr = notes[0]
        for next_note in notes[1:]:
            gap = next_note.start_sec - curr.end_sec
            if next_note.midi_note == curr.midi_note and gap < 0.03:
                # Merge
                # Recalculate duration/end
                curr.end_sec = next_note.end_sec
                # We should average pitch/confidence/velocity ideally, but keeping curr is fine for simple merge
                # Updating velocity to max or avg
                curr.velocity = (curr.velocity + next_note.velocity) / 2
            else:
                merged_notes.append(curr)
                curr = next_note
        merged_notes.append(curr)

    return timeline, merged_notes, []

def _finalize_note_wi(
    notes_list: List[NoteEvent],
    start_time: float,
    end_time: float,
    pitches: List[float],
    confidences: List[float],
    rms_values: List[float],
    min_duration: float
):
    duration = end_time - start_time
    if duration < min_duration:
        return

    median_midi = np.median(pitches)
    rounded_midi = int(round(median_midi))
    avg_conf = float(np.mean(confidences))
    avg_rms = float(np.mean(rms_values))

    pitch_hz = midi_to_hz(rounded_midi)

    note = NoteEvent(
        start_sec=start_time,
        end_sec=end_time,
        midi_note=rounded_midi,
        pitch_hz=pitch_hz,
        confidence=avg_conf,
        velocity=avg_rms # Store raw RMS here, map to MIDI later or here? WI says "Velocity proportional to RMS".
                         # Stage D usually handles formatting, but NoteEvent velocity is usually 0-1.
                         # We'll normalize later or just store raw.
                         # NoteEvent default is 0.8. Let's store raw RMS (usually 0-1ish).
    )
    notes_list.append(note)

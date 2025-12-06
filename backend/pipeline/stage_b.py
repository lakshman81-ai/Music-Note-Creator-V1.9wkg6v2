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
    Stage B: Pitch tracking and note segmentation using Hysteresis (Method B).
    """
    hop_length = meta.hop_length

    # 1. Pitch Tracking
    time_points = None
    f0 = None
    confidence = None
    tracker_name = "pyin"

    # Try CREPE
    crepe_success = False
    if use_crepe:
        try:
            import crepe
            sr_crepe = 16000
            y_crepe = librosa.resample(y, orig_sr=sr, target_sr=sr_crepe)
            step_size_ms = (hop_length / sr) * 1000
            time_points, f0, confidence, _ = crepe.predict(
                y_crepe, sr_crepe, viterbi=True, step_size=step_size_ms, verbose=0
            )
            crepe_success = True
            tracker_name = "crepe"
        except ImportError:
            pass
        except Exception as e:
            print(f"CREPE failed: {e}. Falling back to pyin.")

    if not crepe_success:
        tracker_name = "pyin"
        fmin = librosa.note_to_hz("C1")  # ~32 Hz
        fmax = librosa.note_to_hz("C7")  # ~2093 Hz

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, fill_na=0.0
        )
        time_points = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        confidence = voiced_probs

    # 2. Build Timeline
    timeline: List[FramePitch] = []

    # Pre-calculate midi values for segmentation
    midi_trace = []

    for t, f, c in zip(time_points, f0, confidence):
        p_hz = float(f)
        if np.isnan(p_hz) or p_hz < 20.0:
            p_hz = 0.0
            m_val = None
        else:
            m_val = hz_to_midi(p_hz)

        timeline.append(FramePitch(
            time=float(t),
            pitch_hz=p_hz,
            midi=int(round(m_val)) if m_val is not None else None,
            confidence=float(c)
        ))

        # Store exact midi float for analysis
        midi_trace.append(m_val)

    # 3. Note Segmentation (Hysteresis Method)
    # Rules:
    # - Start note if confidence > high_thresh AND pitch is stable.
    # - Continue note if confidence > low_thresh AND pitch is close to current note pitch.
    # - End note if confidence < low_thresh OR pitch jumps.

    notes: List[NoteEvent] = []

    # Parameters
    high_conf_thresh = 0.6
    low_conf_thresh = 0.3 # Hysteresis floor

    # Pitch jump
    pitch_jump_thresh = 0.7 # semitones

    # State
    active_note_start = None
    active_pitches = []
    active_confidences = []
    active_pitch_ref = None # The pitch we are tracking (median of recent)

    min_duration = 0.06 # 60ms

    for i, frame in enumerate(timeline):
        curr_midi = midi_trace[i]
        curr_conf = frame.confidence

        if active_note_start is None:
            # Look for start
            if curr_midi is not None and curr_conf >= high_conf_thresh:
                active_note_start = frame.time
                active_pitches = [curr_midi]
                active_confidences = [curr_conf]
                active_pitch_ref = curr_midi
        else:
            # Check for continue
            should_continue = False

            if curr_midi is not None:
                # 1. Confidence check
                if curr_conf >= low_conf_thresh:
                    # 2. Pitch jump check
                    # Compare against running average or reference
                    # Let's compare against the *reference* which locks the note intent
                    # But we should allow vibrato.
                    if abs(curr_midi - active_pitch_ref) < pitch_jump_thresh:
                        should_continue = True

                        # Update reference slowly? Or keep locked?
                        # Keeping locked prevents "walking"
                        # But updating allows glissando?
                        # Requirement says "segment based on pitch changes", implying we split on gliss.
                        # So locking is better.
                    else:
                        # Pitch jump!
                        should_continue = False
                else:
                    # Confidence dropped
                    should_continue = False
            else:
                # Unvoiced
                should_continue = False

            if should_continue:
                active_pitches.append(curr_midi)
                active_confidences.append(curr_conf)
            else:
                # End Note
                end_time = frame.time
                # (Or strictly, the previous frame time + duration? Frame time is center usually)
                # Let's say end is this frame's time (gap starts here)

                _finalize_note_hyst(notes, active_note_start, end_time, active_pitches, active_confidences, min_duration)

                # Try to restart immediately?
                # If pitch jumped, this frame is valid for a new note if confidence is high
                active_note_start = None
                active_pitches = []
                active_confidences = []
                active_pitch_ref = None

                if curr_midi is not None and curr_conf >= high_conf_thresh:
                    active_note_start = frame.time
                    active_pitches = [curr_midi]
                    active_confidences = [curr_conf]
                    active_pitch_ref = curr_midi

    # Finalize last note
    if active_note_start is not None:
        _finalize_note_hyst(notes, active_note_start, timeline[-1].time, active_pitches, active_confidences, min_duration)

    chords: List[ChordEvent] = []

    # Store tracker info in meta or return it?
    # The caller expects specific returns.
    # We can't easily return tracker name here unless we change signature, but we can set it in meta if passed by ref.
    # But meta is a dataclass instance.
    # We'll just return standard things. The AnalysisData builder in Orchestrator can handle "pitch_tracker" if we passed it back.
    # But simpler: the orchestrator knows what it asked for.

    return timeline, notes, chords

def _finalize_note_hyst(
    notes_list: List[NoteEvent],
    start_time: float,
    end_time: float,
    pitches: List[float],
    confidences: List[float],
    min_duration: float
):
    duration = end_time - start_time
    if duration < min_duration:
        return

    median_midi = np.median(pitches)
    rounded_midi = int(round(median_midi))
    avg_conf = float(np.mean(confidences))

    pitch_hz = midi_to_hz(rounded_midi)

    note = NoteEvent(
        start_sec=start_time,
        end_sec=end_time,
        midi_note=rounded_midi,
        pitch_hz=pitch_hz,
        confidence=avg_conf
    )
    notes_list.append(note)

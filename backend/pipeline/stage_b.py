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
    Stage B: Pitch tracking, RMS calculation, and note segmentation using Hysteresis (Method B).
    """
    hop_length = meta.hop_length

    # 1. Pitch Tracking
    time_points = None
    f0 = None
    confidence = None

    # Calculate RMS per frame
    rms_frames = librosa.feature.rms(y=y, frame_length=meta.window_size, hop_length=hop_length)[0]

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
        except ImportError:
            pass
        except Exception as e:
            print(f"CREPE failed: {e}. Falling back to pyin.")

    if not crepe_success:
        fmin = librosa.note_to_hz("C1")  # ~32 Hz
        fmax = librosa.note_to_hz("C7")  # ~2093 Hz

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, fill_na=0.0
        )
        time_points = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        confidence = voiced_probs

    # 1.5. Median Pitch Smoothing (Window 11)
    # Apply to raw f0 before processing (ignoring 0/NaN for median usually requires care, but 0 is usually unvoiced)
    # However, standard median filter treats 0 as a value.
    # If unvoiced is 0, median filter might smear silence into pitch or vice versa.
    # But requirement says "Median pitch smoothing (window 11) ... before segmentation".
    # librosa.pyin has already done some Viterbi smoothing.
    # Let's apply scipy.signal.medfilt on the f0 array.
    f0_smoothed = scipy.signal.medfilt(f0, kernel_size=11)

    # 2. Build Timeline
    timeline: List[FramePitch] = []

    # Pre-calculate midi values
    midi_trace = []

    # Ensure rms_frames aligns with f0 length (librosa.feature.rms centering might differ slightly)
    # Usually librosa aligns them if hop_length is same.
    min_len = min(len(time_points), len(f0_smoothed), len(rms_frames))

    for i in range(min_len):
        t = time_points[i]
        p_hz = float(f0_smoothed[i])
        c = confidence[i]
        r = float(rms_frames[i])

        if np.isnan(p_hz) or p_hz < 20.0:
            p_hz = 0.0
            m_val = None
        else:
            m_val = hz_to_midi(p_hz)

        timeline.append(FramePitch(
            time=float(t),
            pitch_hz=p_hz,
            midi=int(round(m_val)) if m_val is not None else None,
            confidence=float(c),
            rms=r
        ))
        midi_trace.append(m_val)

    # 3. Note Segmentation (Hysteresis Method + 3-frame Stability)
    notes: List[NoteEvent] = []

    # Parameters
    high_conf_thresh = 0.6
    low_conf_thresh = 0.3
    pitch_jump_thresh = 0.7  # semitones (tolerance)
    min_frames_stable = 3

    # State
    active_start_idx = None
    active_ref_midi = None
    active_frame_indices = [] # keep track of indices to sum RMS later

    # For stability check: we need to look ahead or keep buffer?
    # Actually, we can just iterate.
    # A note starts if we have N consecutive frames > high_thresh

    # We'll use a state machine with a buffer for potential start.
    potential_start_buffer = []

    i = 0
    while i < len(timeline):
        frame = timeline[i]
        curr_midi = midi_trace[i]
        curr_conf = frame.confidence

        if active_start_idx is None:
            # LOOKING FOR START
            # Check if this frame is valid candidate
            if curr_midi is not None and curr_conf >= high_conf_thresh:
                potential_start_buffer.append(i)
            else:
                potential_start_buffer.clear()

            # Check stability rule
            if len(potential_start_buffer) >= min_frames_stable:
                # We have enough stable frames to confirm start.
                # Check pitch stability within the buffer?
                # "pitch stability" - let's ensure they are close to each other.
                # Take the first frame in buffer as reference, or average?
                # Let's take the first frame's pitch as reference candidate.
                first_idx = potential_start_buffer[0]
                ref_midi = midi_trace[first_idx]

                # Check if all in buffer are close to ref
                is_stable = True
                for idx in potential_start_buffer[1:]:
                    if abs(midi_trace[idx] - ref_midi) > pitch_jump_thresh:
                        is_stable = False
                        break

                if is_stable:
                    # START NOTE
                    active_start_idx = first_idx
                    active_ref_midi = ref_midi # Locking pitch to start
                    active_frame_indices = list(potential_start_buffer)
                    potential_start_buffer = [] # Clear buffer, we are now in active mode
                else:
                    # Not stable, shift buffer (remove first, keep others)
                    # Ideally we slide window.
                    potential_start_buffer.pop(0)

        else:
            # NOTE ACTIVE
            should_continue = False

            if curr_midi is not None and curr_conf >= low_conf_thresh:
                 # Check pitch proximity to LOCKED reference
                 if abs(curr_midi - active_ref_midi) < pitch_jump_thresh:
                     should_continue = True

            if should_continue:
                active_frame_indices.append(i)
            else:
                # END NOTE
                _finalize_note(notes, timeline, active_frame_indices, active_ref_midi)

                # Reset
                active_start_idx = None
                active_ref_midi = None
                active_frame_indices = []
                potential_start_buffer = []

                # Re-evaluate current frame for new start?
                # Yes, but we need 3 frames including this one.
                # The loop continues, next iteration i+1.
                # But we might miss this frame as start of next note.
                # We should re-process this frame in "LOOKING FOR START" state.
                # Decrement i to retry this frame?
                # Yes.
                i -= 1

        i += 1

    # Finalize if active
    if active_start_idx is not None:
         _finalize_note(notes, timeline, active_frame_indices, active_ref_midi)

    chords: List[ChordEvent] = []
    return timeline, notes, chords


def _finalize_note(
    notes_list: List[NoteEvent],
    timeline: List[FramePitch],
    indices: List[int],
    locked_midi: float
):
    if not indices:
        return

    start_time = timeline[indices[0]].time
    end_time = timeline[indices[-1]].time

    # Calculate average RMS
    # RMS is linear amplitude (usually 0-1ish for normalized audio)
    rms_values = [timeline[i].rms for i in indices]
    avg_rms = np.mean(rms_values) if rms_values else 0.0

    # Confidence average
    conf_values = [timeline[i].confidence for i in indices]
    avg_conf = np.mean(conf_values) if conf_values else 0.0

    rounded_midi = int(round(locked_midi))
    pitch_hz = midi_to_hz(rounded_midi)

    # Note duration check? (Stage D might handle short notes, or we filter here?)
    # "Gap merging ... < 1/32" is Stage D.
    # Feature extraction usually keeps everything valid.

    note = NoteEvent(
        start_sec=start_time,
        end_sec=end_time,
        midi_note=rounded_midi,
        pitch_hz=pitch_hz,
        confidence=float(avg_conf),
        rms_value=float(avg_rms),
        velocity=0.0 # Will be mapped in Stage D
    )
    notes_list.append(note)

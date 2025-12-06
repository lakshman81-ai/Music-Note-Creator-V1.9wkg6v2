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
    hop_length = meta.hop_length # 512 typically

    # High-res RMS for silence detection
    # Use a small window to catch short dips (e.g. 10ms ~ 220 samples)
    rms_frame_length = 256
    rms_hop_length = 64 # Overlap heavily
    rms = librosa.feature.rms(y=y, frame_length=rms_frame_length, hop_length=rms_hop_length, center=True)[0]

    # Normalize RMS
    if np.max(rms) > 0:
        rms_norm = rms / np.max(rms)
    else:
        rms_norm = rms

    rms_thresh = 0.05

    # Helper to get min RMS in a time range (samples)
    def get_min_rms(start_sample, end_sample):
        start_idx = start_sample // rms_hop_length
        end_idx = end_sample // rms_hop_length
        if start_idx >= len(rms_norm): return 0.0
        if end_idx >= len(rms_norm): end_idx = len(rms_norm)
        if start_idx == end_idx: return rms_norm[start_idx]
        return np.min(rms_norm[start_idx:end_idx])

    # 1. Pitch Tracking
    time_points = None
    f0 = None
    confidence = None

    crepe_success = False
    if use_crepe:
        try:
            import crepe
            sr_crepe = 16000
            y_crepe = librosa.resample(y, orig_sr=sr, target_sr=sr_crepe)
            step_size_ms = (hop_length / sr) * 1000
            time_points, f0, confidence, _ = crepe.predict(y_crepe, sr_crepe, viterbi=True, step_size=step_size_ms, verbose=0)
            crepe_success = True
        except:
            pass

    if not crepe_success:
        fmin = librosa.note_to_hz("C2")
        fmax = librosa.note_to_hz("C7")
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, fill_na=0.0)
        time_points = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        confidence = voiced_probs

    # 2. Build Timeline
    timeline: List[FramePitch] = []

    for i, t in enumerate(time_points):
        # Determine sample range for this frame
        start_samp = i * hop_length
        end_samp = (i + 1) * hop_length

        # Check high-res RMS
        min_r = get_min_rms(start_samp, end_samp)

        f = f0[i] if i < len(f0) else 0.0
        c = confidence[i] if i < len(confidence) else 0.0
        p_hz = float(f)

        # Gating
        if np.isnan(p_hz) or p_hz < 10.0 or min_r < rms_thresh:
            p_hz = 0.0
            midi_val = None
        else:
            midi_val = int(round(hz_to_midi(p_hz)))

        timeline.append(FramePitch(time=float(t), pitch_hz=p_hz, midi=midi_val, confidence=float(c)))

    # 3. Note Segmentation
    notes: List[NoteEvent] = []

    current_start_time = None
    current_midi_values = []
    current_confidences = []

    min_duration = 0.06
    pitch_change_thresh = 0.7

    for frame in timeline:
        is_voiced = frame.midi is not None and frame.pitch_hz > 0

        if is_voiced:
            curr_midi_float = hz_to_midi(frame.pitch_hz)

            if current_start_time is None:
                current_start_time = frame.time
                current_midi_values = [curr_midi_float]
                current_confidences = [frame.confidence]
            else:
                last_midi_float = current_midi_values[-1]
                avg_midi = np.mean(current_midi_values)

                # Check for jump or drift
                if abs(curr_midi_float - last_midi_float) > pitch_change_thresh or \
                   abs(curr_midi_float - avg_midi) > pitch_change_thresh:
                       _finalize_note(notes, current_start_time, frame.time, current_midi_values, current_confidences, min_duration)
                       current_start_time = frame.time
                       current_midi_values = [curr_midi_float]
                       current_confidences = [frame.confidence]
                else:
                       current_midi_values.append(curr_midi_float)
                       current_confidences.append(frame.confidence)

        else:
            if current_start_time is not None:
                _finalize_note(notes, current_start_time, frame.time, current_midi_values, current_confidences, min_duration)
                current_start_time = None
                current_midi_values = []
                current_confidences = []

    if current_start_time is not None and len(current_midi_values) > 0:
        end_time = timeline[-1].time
        _finalize_note(notes, current_start_time, end_time, current_midi_values, current_confidences, min_duration)

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

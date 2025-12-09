from typing import List, Optional, Set, Dict, Tuple
import numpy as np
import librosa
import scipy.signal
from dataclasses import dataclass
from .models import NoteEvent, AnalysisData, FramePitch, MetaData, AlternativePitch

# Constants for Stage C
HMM_STATE_SILENCE = 0
HMM_STATE_ATTACK = 1
HMM_STATE_STABLE = 2
HMM_NUM_STATES = 3

def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

# --------------------------------------------------------------------------
# HMM Processor for Monophonic Stems (Vocals/Bass)
# --------------------------------------------------------------------------

class HMMProcessor:
    """
    Implements Hidden Markov Model decoding for pitch tracking as per Report Section 7.
    States: Silence (0), Attack (1), Stable (2).
    """
    def __init__(self, fps: float = 100.0):
        self.fps = fps
        # Transition Matrix (A) - Tuned for ~100Hz frame rate (10ms)
        # S0 -> S0: Sticky (0.99)
        # S0 -> SA: 0.01 (Attack)
        # SA -> SS: 0.90 (Quick transition to stable)
        # SA -> S0: 0.10 (False alarm)
        # SS -> SS: 0.99 (Sustain)
        # SS -> S0: 0.01 (Release)

        # Matrix shape (n_states, n_states) -> A[i, j] = P(i -> j)
        # Librosa viterbi expects transition matrix where A[i, j] is prob of going FROM i TO j

        self.trans_mat = np.array([
            [0.99, 0.01, 0.00], # From Silence
            [0.10, 0.00, 0.90], # From Attack (Self transition 0? Attack is transient)
            [0.05, 0.00, 0.95]  # From Stable
        ])

        # Normalization
        self.trans_mat = self.trans_mat / self.trans_mat.sum(axis=1, keepdims=True)

        # Initial probabilities (Priors)
        self.start_prob = np.array([0.9, 0.1, 0.0]) # Bias towards silence at start

    def decode(self, timeline: List[FramePitch]) -> List[int]:
        """
        Runs Viterbi decoding on the timeline.
        Returns state sequence.
        """
        n_frames = len(timeline)
        if n_frames == 0:
            return []

        # Construction Emission Matrix (B)
        # Observation is implicitly the 'confidence' and 'pitch stability'.
        # However, generic Viterbi takes P(Observation | State).
        # We model this directly.

        # P(Obs | Silence): High if confidence is low.
        # P(Obs | Attack): High if high spectral flux? Or just intermediate confidence.
        # P(Obs | Stable): High if confidence is high.

        # Let's use SwiftF0 confidence 'conf'
        # emission[state, time]
        emission = np.zeros((HMM_NUM_STATES, n_frames))

        for t, fp in enumerate(timeline):
            conf = fp.confidence

            # Heuristic mapping based on report
            # Silence favors low confidence
            emission[HMM_STATE_SILENCE, t] = self._prob_silence(conf)

            # Attack favors rising confidence or onset?
            # Report says "Attack allows high spectral flux".
            # For simplicity using just confidence: Attack is a bridge.
            emission[HMM_STATE_ATTACK, t] = self._prob_attack(conf)

            # Stable favors high confidence
            emission[HMM_STATE_STABLE, t] = self._prob_stable(conf)

        # Run Viterbi
        # librosa.sequence.viterbi expects (n_states, n_frames) prob matrix
        state_seq = librosa.sequence.viterbi(emission, self.trans_mat, p_init=self.start_prob)
        return state_seq

    def _prob_silence(self, conf: float) -> float:
        # P(conf | Silence) ~ exp(-conf)
        # If conf=0, p=1. If conf=1, p=small
        return np.exp(-5.0 * conf)

    def _prob_stable(self, conf: float) -> float:
        # P(conf | Stable) ~ sigmoid(conf)
        # If conf=1, p=high
        return 1.0 / (1.0 + np.exp(-10.0 * (conf - 0.5)))

    def _prob_attack(self, conf: float) -> float:
        # Intermediate/Uncertainty
        return 0.3 # Uniform prior for attack frames?

    def segment_notes(self, timeline: List[FramePitch], states: List[int]) -> List[NoteEvent]:
        """
        Converts state sequence back to NoteEvents.
        Logic: A contiguous block of Stable (2) states is a note.
        Attack (1) states preceding Stable are prepended to the note.
        """
        notes = []
        current_note_start = -1
        current_midi_sum = 0.0
        current_hz_sum = 0.0
        current_count = 0
        current_conf_sum = 0.0

        for t, state in enumerate(states):
            is_note_state = (state == HMM_STATE_STABLE) or (state == HMM_STATE_ATTACK)

            if is_note_state:
                if current_note_start == -1:
                    current_note_start = t

                # Accumulate statistics for pitch estimation
                # Use only STABLE states for pitch averaging to avoid onset bending?
                # Or use all. Report says "Stable state demands high pitch stability".
                # Let's assume timeline[t].pitch_hz is valid if confidence > 0
                if timeline[t].pitch_hz > 0:
                     current_hz_sum += timeline[t].pitch_hz
                     current_midi_sum += hz_to_midi(timeline[t].pitch_hz)
                     current_conf_sum += timeline[t].confidence
                     current_count += 1
            else:
                # Silence state
                if current_note_start != -1:
                    # Note ended
                    self._create_note(notes, timeline, current_note_start, t, current_midi_sum, current_hz_sum, current_conf_sum, current_count)
                    current_note_start = -1
                    current_midi_sum = 0.0
                    current_hz_sum = 0.0
                    current_count = 0
                    current_conf_sum = 0.0

        # Close open note
        if current_note_start != -1:
             self._create_note(notes, timeline, current_note_start, len(states), current_midi_sum, current_hz_sum, current_conf_sum, current_count)

        return notes

    def _create_note(self, notes, timeline, start_idx, end_idx, m_sum, h_sum, c_sum, count):
        if count == 0: return # Ghost note (only unvoiced frames?)

        avg_midi = m_sum / count
        avg_hz = h_sum / count
        avg_conf = c_sum / count

        # Round midi
        midi_note = int(round(avg_midi))

        # Refine Hz based on rounded MIDI (standard) or keep expressive pitch?
        # Standard notation requires quantized pitch, but we can store precise.

        start_t = timeline[start_idx].time
        end_t = timeline[end_idx-1].time if end_idx > 0 else start_t + 0.01

        notes.append(NoteEvent(
            start_sec=start_t,
            end_sec=end_t,
            midi_note=midi_note,
            pitch_hz=avg_hz,
            confidence=avg_conf,
            rms_value=0.5 # Placeholder
        ))

# --------------------------------------------------------------------------
# Greedy Processor for Polyphonic Stems (Other)
# --------------------------------------------------------------------------

class GreedyPolyProcessor:
    """
    Enhanced Greedy Tracker for Polyphony (ISS output).
    Uses hysteresis thresholds for note onset/offset.
    """
    def __init__(self):
        self.active_tracks = []
        self.min_duration_frames = 5 # 50ms approx
        self.match_tolerance = 0.7 # Semitones

    def process(self, timeline: List[FramePitch]) -> List[NoteEvent]:
        # 'timeline' here is a list where active_pitches contains ALL concurrent pitches
        # This processor expects timeline[i].active_pitches to be populated

        notes = []

        # State: List of dicts {'midi': float, 'start_t': float, 'conf_accum': float, 'count': int, 'last_seen': int}
        active_tracks = []

        for t, frame in enumerate(timeline):
            curr_time = frame.time
            # Candidates: (hz, conf)
            candidates = frame.active_pitches

            # Map candidates to MIDI
            cand_midi = []
            for hz, conf in candidates:
                if hz > 40:
                    cand_midi.append({'midi': hz_to_midi(hz), 'hz': hz, 'conf': conf, 'used': False})

            # Match existing tracks
            # Greedy match to closest
            next_tracks = []

            for track in active_tracks:
                best_match = None
                best_dist = float('inf')

                for c in cand_midi:
                    if c['used']: continue
                    dist = abs(c['midi'] - track['midi'])
                    if dist < self.match_tolerance and dist < best_dist:
                        best_dist = dist
                        best_match = c

                if best_match:
                    # Update track
                    best_match['used'] = True
                    # Weighted average for pitch stability? Or just follow?
                    # Let's follow slowly (EMA)
                    track['midi'] = 0.8 * track['midi'] + 0.2 * best_match['midi']
                    track['hz_accum'] += best_match['hz']
                    track['conf_accum'] += best_match['conf']
                    track['count'] += 1
                    track['last_seen'] = t
                    track['end_t'] = curr_time
                    next_tracks.append(track)
                else:
                    # Track potentially ending
                    # Allow 1-2 frames dropout?
                    if t - track['last_seen'] > 3:
                        # Close track
                        self._finalize_track(notes, track)
                    else:
                        next_tracks.append(track)

            # Spawn new tracks
            for c in cand_midi:
                if not c['used'] and c['conf'] > 0.15: # Threshold from report
                    new_track = {
                        'midi': c['midi'],
                        'start_t': curr_time,
                        'end_t': curr_time,
                        'hz_accum': c['hz'],
                        'conf_accum': c['conf'],
                        'count': 1,
                        'last_seen': t
                    }
                    next_tracks.append(new_track)

            active_tracks = next_tracks

        # Flush
        for track in active_tracks:
            self._finalize_track(notes, track)

        return notes

    def _finalize_track(self, notes, track):
        if track['count'] < self.min_duration_frames:
            return

        avg_hz = track['hz_accum'] / track['count']
        avg_conf = track['conf_accum'] / track['count']
        midi_note = int(round(track['midi']))

        notes.append(NoteEvent(
            start_sec=track['start_t'],
            end_sec=track['end_t'],
            midi_note=midi_note,
            pitch_hz=avg_hz,
            confidence=avg_conf,
            rms_value=0.5
        ))

# --------------------------------------------------------------------------
# Adaptive Tuning
# --------------------------------------------------------------------------

def estimate_global_tuning_offset(notes: List[NoteEvent]) -> float:
    """
    Computes global tuning offset in semitones based on cent histogram.
    Report Section 8.3.
    """
    if not notes:
        return 0.0

    # Collect fractional parts of MIDI pitches from raw Hz
    # We use the raw Hz to find deviation from standard A440 grid
    deviations = []
    for n in notes:
        raw_midi = hz_to_midi(n.pitch_hz)
        # Deviation from nearest integer (-0.5 to 0.5)
        dev = raw_midi - round(raw_midi)
        deviations.append(dev)

    if not deviations:
        return 0.0

    # Histogram peak
    hist, bins = np.histogram(deviations, bins=50, range=(-0.5, 0.5))
    peak_idx = np.argmax(hist)
    center_val = (bins[peak_idx] + bins[peak_idx+1]) / 2.0

    return center_val

def apply_tuning(notes: List[NoteEvent], offset: float):
    """
    Shifts notes by negative offset to align with A440 grid.
    However, MIDI integers are fixed. We apply this to pitch_hz?
    Or do we shift the 'midi_note' decision boundary?
    Usually, we subtract the offset from the raw MIDI float before rounding.
    """
    for n in notes:
        raw_midi = hz_to_midi(n.pitch_hz)
        corrected_midi = raw_midi - offset
        n.midi_note = int(round(corrected_midi))
        # pitch_hz remains the original audio pitch (truth),
        # or should we correct it? "Correcting for detuned recordings".
        # We usually correct the Quantization.
        # So we just update midi_note.

# --------------------------------------------------------------------------
# Main Stage C Logic
# --------------------------------------------------------------------------

def apply_theory(
    unused_input_notes: List[NoteEvent], # Legacy argument, ignored in favor of analysis_data
    analysis_data: AnalysisData,
) -> List[NoteEvent]:
    """
    Stage C: Polyphonic Segmentation & Quantization
    """
    # 1. Routing
    stem_timelines = analysis_data.stem_timelines

    all_notes = []

    # 2. Process Vocals/Bass (HMM)
    hmm_proc = HMMProcessor()

    for stem_name in ['vocals', 'bass']:
        if stem_name in stem_timelines:
            # Run HMM
            track = stem_timelines[stem_name]
            states = hmm_proc.decode(track)
            notes = hmm_proc.segment_notes(track, states)
            all_notes.extend(notes)

    # 3. Process Other (Greedy)
    greedy_proc = GreedyPolyProcessor()

    if 'other' in stem_timelines:
        track = stem_timelines['other']
        # This track has active_pitches populated
        notes = greedy_proc.process(track)
        all_notes.extend(notes)

    # Fallback: If no stems (legacy mode), use main timeline with HMM?
    # Or Greedy if polyphonic?
    if not all_notes and analysis_data.timeline:
        # Check audio type
        is_poly = analysis_data.meta.audio_type != "monophonic"
        if is_poly:
            notes = greedy_proc.process(analysis_data.timeline)
        else:
            states = hmm_proc.decode(analysis_data.timeline)
            notes = hmm_proc.segment_notes(analysis_data.timeline, states)
        all_notes.extend(notes)

    # 4. Adaptive Tuning
    tuning_offset = estimate_global_tuning_offset(all_notes)
    analysis_data.meta.tuning_offset = tuning_offset
    apply_tuning(all_notes, tuning_offset)

    # 5. Quantize
    analysis_data.notes = all_notes
    quantized_notes = quantize_notes(all_notes, analysis_data)

    return quantized_notes

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

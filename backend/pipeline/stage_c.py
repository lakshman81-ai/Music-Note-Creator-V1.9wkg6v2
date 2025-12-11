from typing import List, Optional, Set, Dict, Tuple
import numpy as np
import librosa
import scipy.signal
from .models import NoteEvent, AnalysisData, FramePitch, MetaData, AudioQuality
from .config import PIANO_61KEY_CONFIG, StageCConfig, PipelineConfig

# Constants for HMM
HMM_STATE_SILENCE = 0
HMM_STATE_ATTACK = 1
HMM_STATE_STABLE = 2
HMM_NUM_STATES = 3

def hz_to_midi(hz: float) -> float:
    if hz <= 0: return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

class HMMProcessor:
    def __init__(self, config: StageCConfig):
        self.config = config
        # Transition Probabilities (Could be configurable, but hardcoded robust logic for now)
        self.trans_mat = np.array([
            [0.99, 0.01, 0.00], # From Silence
            [0.10, 0.00, 0.90], # From Attack
            [0.05, 0.00, 0.95]  # From Stable
        ])
        self.trans_mat = self.trans_mat / self.trans_mat.sum(axis=1, keepdims=True)
        self.start_prob = np.array([0.9, 0.1, 0.0])

    def decode(self, timeline: List[FramePitch]) -> List[int]:
        n_frames = len(timeline)
        if n_frames == 0: return []

        emission = np.zeros((HMM_NUM_STATES, n_frames))

        for t, fp in enumerate(timeline):
            conf = fp.confidence
            # Heuristic Emission Probs
            # Silence: High prob if conf low
            emission[HMM_STATE_SILENCE, t] = np.exp(-5.0 * conf)
            # Attack: Transient? We don't have transient info passed here explicitly,
            # but high confidence onset usually looks like attack.
            emission[HMM_STATE_ATTACK, t] = 0.3 # Constant bias
            # Stable: High prob if conf high
            emission[HMM_STATE_STABLE, t] = 1.0 / (1.0 + np.exp(-10.0 * (conf - 0.5)))

        try:
             state_seq = librosa.sequence.viterbi(emission, self.trans_mat, p_init=self.start_prob)
             return state_seq
        except Exception:
             return [HMM_STATE_SILENCE] * n_frames

    def segment_notes(self, timeline: List[FramePitch], states: List[int]) -> List[NoteEvent]:
        notes = []
        current_note_start = -1
        accum_data = {'midi': [], 'hz': [], 'conf': [], 'rms': []}

        min_frames = self.config.frame_stability.get("stable_frames_required", 2)
        min_dur_sec = self.config.min_note_duration_ms / 1000.0

        for t, state in enumerate(states):
            is_note = (state == HMM_STATE_STABLE) or (state == HMM_STATE_ATTACK)

            if is_note:
                if current_note_start == -1:
                    current_note_start = t

                # Frame Stability Check (if just started)
                # (Simple check: if we are in state note, we accumulate)

                fp = timeline[t]
                if fp.pitch_hz > 0:
                     accum_data['midi'].append(hz_to_midi(fp.pitch_hz))
                     accum_data['hz'].append(fp.pitch_hz)
                     accum_data['conf'].append(fp.confidence)
                     accum_data['rms'].append(fp.rms)

                     # LEGATO / PITCH JUMP CHECK
                     if len(accum_data['midi']) > 5:
                         curr_avg = np.mean(accum_data['midi'][-5:])
                         last_avg = np.mean(accum_data['midi'][:-1])
                         if abs(curr_avg - last_avg) > 0.8: # Jump
                             # Split
                             self._close_note(notes, timeline, current_note_start, t, accum_data, min_dur_sec)
                             current_note_start = t
                             accum_data = {'midi': [hz_to_midi(fp.pitch_hz)], 'hz': [fp.pitch_hz], 'conf': [fp.confidence], 'rms': [fp.rms]}

            else:
                if current_note_start != -1:
                    self._close_note(notes, timeline, current_note_start, t, accum_data, min_dur_sec)
                    current_note_start = -1
                    accum_data = {'midi': [], 'hz': [], 'conf': [], 'rms': []}

        if current_note_start != -1:
             self._close_note(notes, timeline, current_note_start, len(states), accum_data, min_dur_sec)

        # Gap Filling (Legato)
        # "If gap_ms <= max_gap_ms, and pitch similar, merge."
        max_gap_sec = self.config.gap_filling.get("max_gap_ms", 100.0) / 1000.0
        merged_notes = []
        if notes:
            merged_notes.append(notes[0])
            for i in range(1, len(notes)):
                curr = notes[i]
                prev = merged_notes[-1]
                gap = curr.start_sec - prev.end_sec
                if gap <= max_gap_sec and abs(curr.midi_note - prev.midi_note) < (self.config.pitch_tolerance_cents / 100.0):
                    # Merge
                    prev.end_sec = curr.end_sec
                    # Update averages? keep prev usually fine.
                else:
                    merged_notes.append(curr)

        return merged_notes

    def _close_note(self, notes, timeline, start, end, data, min_dur_sec):
        dur = timeline[end-1].time - timeline[start].time
        if dur < min_dur_sec: return

        if not data['hz']: return

        avg_hz = np.mean(data['hz'])
        avg_conf = np.mean(data['conf'])
        avg_rms = np.mean(data['rms']) if data['rms'] else 0.0
        midi_note = int(round(hz_to_midi(avg_hz)))

        notes.append(NoteEvent(
            start_sec=timeline[start].time,
            end_sec=timeline[end-1].time,
            midi_note=midi_note,
            pitch_hz=float(avg_hz),
            confidence=float(avg_conf),
            rms_value=float(avg_rms)
        ))


class GreedyPolyProcessor:
    def __init__(self, config: StageCConfig):
        self.config = config
        self.active_tracks = []
        self.match_tolerance = 0.7

    def process(self, timeline: List[FramePitch]) -> List[NoteEvent]:
        notes = []
        active_tracks = []
        min_dur_sec = self.config.min_note_duration_ms / 1000.0

        for t, frame in enumerate(timeline):
            curr_time = frame.time
            candidates = frame.active_pitches # [(hz, conf), ...]
            cand_midi = []
            for hz, conf in candidates:
                if hz > 40:
                    cand_midi.append({'midi': hz_to_midi(hz), 'hz': hz, 'conf': conf, 'used': False})

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
                    best_match['used'] = True
                    # Update track state
                    track['midi'] = 0.8 * track['midi'] + 0.2 * best_match['midi']
                    track['hz_accum'].append(best_match['hz'])
                    track['conf_accum'].append(best_match['conf'])
                    track['end_t'] = curr_time
                    track['last_seen'] = t
                    next_tracks.append(track)
                else:
                    # Lost track
                    if t - track['last_seen'] > 3: # 3 frames tolerance dropout
                        self._finalize_track(notes, track, min_dur_sec)
                    else:
                        next_tracks.append(track)

            # Spawn new
            for c in cand_midi:
                if not c['used'] and c['conf'] > 0.15: # Start threshold
                    new_track = {
                        'midi': c['midi'],
                        'start_t': curr_time,
                        'end_t': curr_time,
                        'hz_accum': [c['hz']],
                        'conf_accum': [c['conf']],
                        'last_seen': t
                    }
                    next_tracks.append(new_track)

            active_tracks = next_tracks

        for track in active_tracks:
            self._finalize_track(notes, track, min_dur_sec)

        return notes

    def _finalize_track(self, notes, track, min_dur_sec):
        dur = track['end_t'] - track['start_t']
        if dur < min_dur_sec: return

        avg_hz = np.mean(track['hz_accum'])
        avg_conf = np.mean(track['conf_accum'])
        midi_note = int(round(track['midi']))

        notes.append(NoteEvent(
            start_sec=track['start_t'],
            end_sec=track['end_t'],
            midi_note=midi_note,
            pitch_hz=float(avg_hz),
            confidence=float(avg_conf),
            rms_value=0.5 # Placeholder
        ))

# --- Utils ---

def estimate_global_tuning_offset(notes: List[NoteEvent]) -> float:
    if not notes: return 0.0
    deviations = []
    for n in notes:
        raw_midi = hz_to_midi(n.pitch_hz)
        dev = raw_midi - round(raw_midi)
        deviations.append(dev)

    if not deviations: return 0.0
    hist, bins = np.histogram(deviations, bins=50, range=(-0.5, 0.5))
    peak_idx = np.argmax(hist)
    center_val = (bins[peak_idx] + bins[peak_idx+1]) / 2.0
    return center_val

def apply_tuning(notes: List[NoteEvent], offset: float):
    for n in notes:
        raw_midi = hz_to_midi(n.pitch_hz)
        corrected_midi = raw_midi - offset
        n.midi_note = int(round(corrected_midi))

def map_velocity(notes: List[NoteEvent], config: StageCConfig):
    # vel = min_vel + (db - min_db) * (max_vel - min_vel) / (max_db - min_db)
    v_map = config.velocity_map
    min_db, max_db = v_map["min_db"], v_map["max_db"]
    min_v, max_v = v_map["min_vel"], v_map["max_vel"]

    for n in notes:
        # RMS to dB
        if n.rms_value <= 0: db = -80
        else: db = 20 * np.log10(n.rms_value)

        clamped = max(min_db, min(max_db, db))
        ratio = (clamped - min_db) / (max_db - min_db)

        vel = min_v + ratio * (max_v - min_v)
        n.velocity = vel / 127.0 # Normalize 0-1 for model

        # Dynamic marking
        if vel < 45: n.dynamic = "p"
        elif vel < 80: n.dynamic = "mf"
        else: n.dynamic = "f"

# --- Main Stage C ---

def apply_theory(
    analysis_data: AnalysisData,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
) -> List[NoteEvent]:

    stage_c_conf = config.stage_c
    stem_timelines = analysis_data.stem_timelines
    all_notes = []

    # Process Vocals/Bass with HMM (Monophonic)
    hmm_proc = HMMProcessor(stage_c_conf)
    for stem in ['vocals', 'bass']:
        if stem in stem_timelines:
            tl = stem_timelines[stem]
            states = hmm_proc.decode(tl)
            notes = hmm_proc.segment_notes(tl, states)
            for n in notes:
                n.voice = 1 # Lead
                n.staff = "treble" if stem == "vocals" else "bass"
            all_notes.extend(notes)

    # Process Other/Piano with Greedy (Polyphonic)
    greedy_proc = GreedyPolyProcessor(stage_c_conf)
    for stem in ['other', 'piano', 'mix']:
        if stem in stem_timelines:
            tl = stem_timelines[stem]
            notes = greedy_proc.process(tl)
            for n in notes:
                n.voice = 1
                # Staff determined later in Stage D
                # But we can set default
                n.staff = "treble"
            all_notes.extend(notes)

    # Adaptive Tuning
    offset = estimate_global_tuning_offset(all_notes)
    analysis_data.meta.tuning_offset = offset
    apply_tuning(all_notes, offset)

    # Velocity Mapping
    map_velocity(all_notes, stage_c_conf)

    # Quantize (Simple Grid Assignment)
    # Note: Stage D does more complex XML timing, but we assign grid data here.
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    quarter_dur = 60.0 / bpm

    # Align to beats if available
    # For now, simple quantization
    for n in all_notes:
        n.duration_beats = (n.end_sec - n.start_sec) / quarter_dur
        # Note: 'measure' and 'beat' are left None, to be filled by music21 in Stage D or complex logic here.
        # WI says "Output ... Each note a dict ... duration_sec, velocity".
        # Stage D "Turn note events into valid MusicXML ... Sum divisions in a measure".
        # So Stage D handles the measure wrapping.

    analysis_data.notes = all_notes
    return all_notes

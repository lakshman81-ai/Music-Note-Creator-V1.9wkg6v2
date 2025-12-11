from typing import List, Optional, Dict
import numpy as np
import librosa
import scipy.signal

from .models import NoteEvent, AnalysisData, FramePitch
from .config import PIANO_61KEY_CONFIG, StageCConfig, PipelineConfig

# ------------------------------------------------------------
# CONSTANTS / ENUMS
# ------------------------------------------------------------

HMM_STATE_SILENCE = 0
HMM_STATE_ATTACK = 1
HMM_STATE_STABLE = 2
HMM_NUM_STATES = 3


# ------------------------------------------------------------
# UTILS: PITCH CONVERSION
# ------------------------------------------------------------

def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# ------------------------------------------------------------
# HMM-BASED MONOPHONIC SEGMENTATION (VOCALS / BASS)
# ------------------------------------------------------------

class HMMProcessor:
    """
    Implements an HMM over [silence, attack, stable] states for monophonic stems.

    WI alignment:
    - Uses frame_stability.stable_frames_required before confirming note onset.
    - Uses pitch_tolerance_cents + vibrato split threshold to split on real jumps.
    - Applies min_note_duration_ms to prune micro-notes.
    - Uses gap_filling.max_gap_ms and pitch_tolerance_cents to merge legato gaps.
    """

    def __init__(self, config: StageCConfig):
        self.config = config

        # Transition probabilities (Silence -> Attack -> Stable)
        trans = np.array([
            [0.99, 0.01, 0.00],  # Silence
            [0.10, 0.00, 0.90],  # Attack
            [0.05, 0.00, 0.95],  # Stable
        ])
        self.trans_mat = trans / trans.sum(axis=1, keepdims=True)
        self.start_prob = np.array([0.9, 0.1, 0.0])

        # Config-driven thresholds
        self.stable_frames_required: int = (
            self.config.frame_stability.get("stable_frames_required", 2)
        )
        self.pitch_tol_semitones: float = (
            getattr(self.config, "pitch_tolerance_cents", 50.0) / 100.0
        )
        # Vibrato split control (link to SwiftF0 split_semitone if present)
        self.vibrato_split_semitone: float = float(
            getattr(self.config, "vibrato_split_semitone", 0.7)
        )

    # ----------------------------
    # HMM Decode
    # ----------------------------

    def decode(self, timeline: List[FramePitch]) -> List[int]:
        """
        Returns the most probable HMM state sequence for each frame in the timeline.
        Emission is based on frame confidence only (for now).
        """
        n_frames = len(timeline)
        if n_frames == 0:
            return []

        emission = np.zeros((HMM_NUM_STATES, n_frames), dtype=float)

        for t, fp in enumerate(timeline):
            conf = float(fp.confidence)

            # Silence: high if confidence is low
            emission[HMM_STATE_SILENCE, t] = np.exp(-5.0 * conf)

            # Attack: constant bias (could later use onset strength)
            emission[HMM_STATE_ATTACK, t] = 0.3

            # Stable: sigmoid of confidence (high conf → stable)
            emission[HMM_STATE_STABLE, t] = 1.0 / (1.0 + np.exp(-10.0 * (conf - 0.5)))

        try:
            state_seq = librosa.sequence.viterbi(
                emission,
                self.trans_mat,
                p_init=self.start_prob,
            )
            return list(state_seq)
        except Exception:
            # Fallback: everything silence
            return [HMM_STATE_SILENCE] * n_frames

    # ----------------------------
    # Note Segmentation
    # ----------------------------

    def segment_notes(self, timeline: List[FramePitch], states: List[int]) -> List[NoteEvent]:
        """
        Convert HMM state sequence + frame pitches into discrete NoteEvent list.
        Applies:
        - frame stability gate
        - vibrato/jump-based splitting
        - min duration pruning
        - gap-filling merge
        """
        notes: List[NoteEvent] = []
        n_frames = len(timeline)
        if n_frames == 0:
            return notes

        current_note_start: int = -1
        stability_counter: int = 0

        accum_data: Dict[str, List[float]] = {
            "midi": [],
            "hz": [],
            "conf": [],
            "rms": [],
        }

        min_dur_sec: float = self.config.min_note_duration_ms / 1000.0
        max_gap_sec: float = self.config.gap_filling.get("max_gap_ms", 100.0) / 1000.0

        def reset_accum():
            return {"midi": [], "hz": [], "conf": [], "rms": []}

        for t, state in enumerate(states):
            fp = timeline[t]
            is_note_state = state in (HMM_STATE_ATTACK, HMM_STATE_STABLE)

            if is_note_state and fp.pitch_hz > 0.0 and fp.confidence > 0.0:
                # Increment stability counter if we see a valid pitched frame
                stability_counter += 1

                # Only mark onset when we have stable_frames_required consecutive frames
                if current_note_start == -1 and stability_counter >= self.stable_frames_required:
                    # backtrack onset to the first of the stable run
                    current_note_start = max(0, t - self.stable_frames_required + 1)

                # If a note is active, accumulate stats
                if current_note_start != -1:
                    curr_midi = hz_to_midi(fp.pitch_hz)
                    accum_data["midi"].append(curr_midi)
                    accum_data["hz"].append(fp.pitch_hz)
                    accum_data["conf"].append(fp.confidence)
                    accum_data["rms"].append(fp.rms)

                    # Vibrato / jump split control:
                    # observe last small window vs history to detect real pitch jump
                    if len(accum_data["midi"]) > 8:
                        recent = np.array(accum_data["midi"][-5:], dtype=float)
                        history = np.array(accum_data["midi"][:-5], dtype=float)
                        if history.size > 0:
                            curr_avg = float(np.mean(recent))
                            hist_avg = float(np.mean(history))
                            if abs(curr_avg - hist_avg) > self.vibrato_split_semitone:
                                # Close previous note at t-1
                                self._close_note(
                                    notes,
                                    timeline,
                                    current_note_start,
                                    t,
                                    accum_data,
                                    min_dur_sec,
                                )
                                # Start new note at t
                                current_note_start = t
                                accum_data = reset_accum()
                                accum_data["midi"].append(curr_midi)
                                accum_data["hz"].append(fp.pitch_hz)
                                accum_data["conf"].append(fp.confidence)
                                accum_data["rms"].append(fp.rms)
            else:
                # Not in note state: reset stability
                stability_counter = 0

                if current_note_start != -1:
                    self._close_note(
                        notes,
                        timeline,
                        current_note_start,
                        t,
                        accum_data,
                        min_dur_sec,
                    )
                    current_note_start = -1
                    accum_data = reset_accum()

        # Tail case: end with an open note
        if current_note_start != -1:
            self._close_note(
                notes,
                timeline,
                current_note_start,
                len(states),
                accum_data,
                min_dur_sec,
            )

        # Gap filling / legato merge
        merged_notes: List[NoteEvent] = []
        if notes:
            merged_notes.append(notes[0])
            for i in range(1, len(notes)):
                curr = notes[i]
                prev = merged_notes[-1]
                gap = curr.start_sec - prev.end_sec
                if gap <= max_gap_sec:
                    if abs(curr.midi_note - prev.midi_note) <= self.pitch_tol_semitones:
                        # extend previous note
                        prev.end_sec = curr.end_sec
                        # keep previous averages
                    else:
                        merged_notes.append(curr)
                else:
                    merged_notes.append(curr)

        return merged_notes

    # ----------------------------
    # Internal: close note
    # ----------------------------

    def _close_note(
        self,
        notes: List[NoteEvent],
        timeline: List[FramePitch],
        start_idx: int,
        end_idx: int,
        data: Dict[str, List[float]],
        min_dur_sec: float,
    ) -> None:
        if end_idx <= start_idx:
            return

        start_time = timeline[start_idx].time
        end_time = timeline[end_idx - 1].time
        dur = end_time - start_time
        if dur < min_dur_sec:
            return

        if not data["hz"]:
            return

        avg_hz = float(np.mean(data["hz"]))
        avg_conf = float(np.mean(data["conf"])) if data["conf"] else 0.0
        avg_rms = float(np.mean(data["rms"])) if data["rms"] else 0.0
        midi_note = int(round(hz_to_midi(avg_hz)))

        notes.append(
            NoteEvent(
                start_sec=start_time,
                end_sec=end_time,
                midi_note=midi_note,
                pitch_hz=avg_hz,
                confidence=avg_conf,
                rms_value=avg_rms,
            )
        )


# ------------------------------------------------------------
# GREEDY POLYPHONIC TRACKER (OTHER / PIANO / MIX)
# ------------------------------------------------------------

class GreedyPolyProcessor:
    """
    Greedy multi-track tracker over per-frame active pitches.

    WI alignment:
    - Uses config-driven pitch tolerance (pitch_tolerance_cents).
    - Applies min_note_duration_ms when finalizing tracks.
    - Allows dropout frames up to time_merge_tolerance_frames.
    - Outputs polyphonic NoteEvents which can later be filtered (skyline).
    """

    def __init__(self, config: StageCConfig):
        self.config = config
        self.pitch_tol_semitones: float = (
            getattr(self.config, "pitch_tolerance_cents", 50.0) / 100.0
        )
        # frames of allowed dropout before closing track
        self.drop_frames: int = int(
            getattr(self.config, "time_merge_tolerance_frames", 3)
        )

    def process(self, timeline: List[FramePitch]) -> List[NoteEvent]:
        notes: List[NoteEvent] = []
        if not timeline:
            return notes

        active_tracks: List[Dict] = []
        min_dur_sec: float = self.config.min_note_duration_ms / 1000.0

        for frame_idx, frame in enumerate(timeline):
            curr_time = frame.time
            candidates = frame.active_pitches  # list[(hz, conf), ...]
            cand_midi = []

            for hz, conf in candidates:
                if hz > 40.0 and conf > 0.0:
                    cand_midi.append(
                        {
                            "midi": hz_to_midi(hz),
                            "hz": hz,
                            "conf": conf,
                            "used": False,
                        }
                    )

            next_tracks: List[Dict] = []

            # Update existing tracks
            for track in active_tracks:
                best_match = None
                best_dist = float("inf")

                for c in cand_midi:
                    if c["used"]:
                        continue
                    dist = abs(c["midi"] - track["midi"])
                    if dist < self.pitch_tol_semitones and dist < best_dist:
                        best_dist = dist
                        best_match = c

                if best_match is not None:
                    best_match["used"] = True
                    # Simple smoothing of track midi center
                    track["midi"] = 0.8 * track["midi"] + 0.2 * best_match["midi"]
                    track["hz_accum"].append(best_match["hz"])
                    track["conf_accum"].append(best_match["conf"])
                    track["end_t"] = curr_time
                    track["last_seen_idx"] = frame_idx
                    next_tracks.append(track)
                else:
                    # Track not matched this frame
                    if frame_idx - track["last_seen_idx"] > self.drop_frames:
                        self._finalize_track(notes, track, min_dur_sec)
                    else:
                        next_tracks.append(track)

            # Spawn tracks for leftover candidates
            for c in cand_midi:
                if not c["used"] and c["conf"] > 0.15:  # start threshold
                    new_track = {
                        "midi": c["midi"],
                        "start_t": curr_time,
                        "end_t": curr_time,
                        "hz_accum": [c["hz"]],
                        "conf_accum": [c["conf"]],
                        "last_seen_idx": frame_idx,
                    }
                    next_tracks.append(new_track)

            active_tracks = next_tracks

        # Finalize still-open tracks
        for track in active_tracks:
            self._finalize_track(notes, track, min_dur_sec)

        return notes

    def _finalize_track(self, notes: List[NoteEvent], track: Dict, min_dur_sec: float) -> None:
        dur = track["end_t"] - track["start_t"]
        if dur < min_dur_sec:
            return

        avg_hz = float(np.mean(track["hz_accum"]))
        avg_conf = float(np.mean(track["conf_accum"]))
        midi_note = int(round(track["midi"]))

        notes.append(
            NoteEvent(
                start_sec=track["start_t"],
                end_sec=track["end_t"],
                midi_note=midi_note,
                pitch_hz=avg_hz,
                confidence=avg_conf,
                rms_value=0.5,  # Stage B or RMS aggregation can refine this later
            )
        )


# ------------------------------------------------------------
# SKYLINE FILTER (TOP-VOICE EXTRACTION)
# ------------------------------------------------------------

def apply_skyline_filter(notes: List[NoteEvent]) -> List[NoteEvent]:
    """
    Skyline (top-voice) filter: for any overlapping region in time,
    keep only the highest MIDI note (melody line).

    WI alignment:
    - Polyphony_filter.mode == "skyline_top_voice" selects dominant melody
      from polyphonic transcriptions.
    """
    if not notes:
        return []

    # Sort by start time, then pitch
    notes_sorted = sorted(notes, key=lambda n: (n.start_sec, n.midi_note))

    skyline: List[NoteEvent] = []
    for n in notes_sorted:
        if not skyline:
            skyline.append(n)
            continue

        last = skyline[-1]
        # Overlap in time?
        if n.start_sec < last.end_sec:
            # Keep whichever has higher pitch
            if n.midi_note > last.midi_note:
                skyline[-1] = n
        else:
            skyline.append(n)

    return skyline


# ------------------------------------------------------------
# GLOBAL TUNING OFFSET & VELOCITY
# ------------------------------------------------------------

def estimate_global_tuning_offset(notes: List[NoteEvent]) -> float:
    if not notes:
        return 0.0
    deviations: List[float] = []

    for n in notes:
        raw_midi = hz_to_midi(n.pitch_hz)
        dev = raw_midi - round(raw_midi)
        deviations.append(dev)

    if not deviations:
        return 0.0

    hist, bins = np.histogram(deviations, bins=50, range=(-0.5, 0.5))
    peak_idx = int(np.argmax(hist))
    center_val = (bins[peak_idx] + bins[peak_idx + 1]) / 2.0
    return float(center_val)


def apply_tuning(notes: List[NoteEvent], offset: float) -> None:
    for n in notes:
        raw_midi = hz_to_midi(n.pitch_hz)
        corrected_midi = raw_midi - offset
        n.midi_note = int(round(corrected_midi))


def map_velocity(notes: List[NoteEvent], config: StageCConfig) -> None:
    """
    RMS → MIDI velocity, then normalized 0–1.
    WI alignment:
    - min_db, max_db, min_vel, max_vel from config.velocity_map.
    - assigns dynamic marking (p, mf, f).
    """
    v_map = config.velocity_map
    min_db = float(v_map["min_db"])
    max_db = float(v_map["max_db"])
    min_v = float(v_map["min_vel"])
    max_v = float(v_map["max_vel"])

    for n in notes:
        if n.rms_value <= 0.0:
            db = -80.0
        else:
            db = float(20.0 * np.log10(n.rms_value))

        clamped = max(min_db, min(max_db, db))
        ratio = (clamped - min_db) / (max_db - min_db + 1e-9)
        vel = min_v + ratio * (max_v - min_v)

        n.velocity = vel / 127.0  # normalized 0–1 for model
        if vel < 45:
            n.dynamic = "p"
        elif vel < 80:
            n.dynamic = "mf"
        else:
            n.dynamic = "f"


# ------------------------------------------------------------
# QUANTIZATION UTIL
# ------------------------------------------------------------

def _parse_grid_step(grid_str: str) -> float:
    """
    Parse a grid like '1/16' or '1/32' to beat step size.
    '1/4' = quarter note = 1.0 beat, '1/8' = 0.5, '1/16' = 0.25, ...
    """
    if not grid_str or "/" not in grid_str:
        return 0.25  # default to 1/16

    num, den = grid_str.split("/")
    try:
        num = float(num)
        den = float(den)
        # beats per note of that fraction if 1 beat = quarter note
        return (4.0 * num) / den
    except Exception:
        return 0.25


def quantize_notes_to_grid(
    notes: List[NoteEvent],
    bpm: float,
    quantization_grid: Dict[str, str],
) -> None:
    """
    Assign duration_beats and (optionally) quantized start beats to each note,
    based on a configurable grid.

    WI alignment:
    - Stage C: "Quantization Grid: 1/16 or 1/32 beat grid".
    - Stage D then turns these into divisions / measures.
    """
    if not notes:
        return

    primary_grid = quantization_grid.get("primary", "1/16")
    fast_grid = quantization_grid.get("fast_passages", primary_grid)

    primary_step = _parse_grid_step(primary_grid)
    fast_step = _parse_grid_step(fast_grid)

    quarter_dur = 60.0 / bpm

    for n in notes:
        dur_sec = n.end_sec - n.start_sec
        raw_beats = dur_sec / quarter_dur

        # If very short note → use fast grid
        grid_step = fast_step if raw_beats <= primary_step else primary_step

        # Quantize duration and (optionally) onset
        q_dur_beats = round(raw_beats / grid_step) * grid_step
        q_dur_beats = max(q_dur_beats, grid_step)  # avoid zero

        start_beats = n.start_sec / quarter_dur
        q_start_beats = round(start_beats / grid_step) * grid_step

        n.duration_beats = q_dur_beats
        # Stage D can use q_start_beats if needed:
        n.start_beats = q_start_beats


# ------------------------------------------------------------
# MAIN STAGE C ENTRYPOINT
# ------------------------------------------------------------

def apply_theory(
    analysis_data: AnalysisData,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
) -> List[NoteEvent]:
    """
    Stage C: Note tracking / segmentation.

    Steps:
    - HMM mono processing for vocals, bass.
    - Greedy poly processing for other/piano/mix.
    - Skyline filter (optional, based on config).
    - Global tuning offset estimation and application.
    - RMS → velocity mapping.
    - Quantize to beat grid (duration_beats, start_beats).
    """
    stage_c_conf: StageCConfig = config.stage_c
    stem_timelines = analysis_data.stem_timelines
    all_notes: List[NoteEvent] = []

    # --- HMM for monophonic stems (vocals, bass) ---
    hmm_proc = HMMProcessor(stage_c_conf)
    for stem in ["vocals", "bass"]:
        if stem in stem_timelines:
            tl = stem_timelines[stem]
            states = hmm_proc.decode(tl)
            notes = hmm_proc.segment_notes(tl, states)
            for n in notes:
                n.voice = 1
                n.staff = "treble" if stem == "vocals" else "bass"
            all_notes.extend(notes)

    # --- Greedy polyphonic tracker for piano/other/mix ---
    greedy_proc = GreedyPolyProcessor(stage_c_conf)
    for stem in ["other", "piano", "mix"]:
        if stem in stem_timelines:
            tl = stem_timelines[stem]
            notes = greedy_proc.process(tl)
            for n in notes:
                n.voice = 1
                # staff assignment deferred to Stage D (C4 split), default to treble here
                n.staff = "treble"
            all_notes.extend(notes)

    # --- Optional skyline filter (top-voice extraction) ---
    poly_filter = getattr(stage_c_conf, "polyphony_filter", {})
    mode = poly_filter.get("mode", "").lower()
    if mode == "skyline_top_voice":
        all_notes = apply_skyline_filter(all_notes)

    # --- Global tuning ---
    offset = estimate_global_tuning_offset(all_notes)
    analysis_data.meta.tuning_offset = offset
    apply_tuning(all_notes, offset)

    # --- Velocity mapping ---
    map_velocity(all_notes, stage_c_conf)

    # Quantize (Simple Grid Assignment)
    # Note: Stage D does more complex XML timing, but we assign grid data here.
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0

    # Some upstream code (e.g., librosa beat tracking) can return numpy scalar/array.
    # Ensure bpm is a plain float to avoid numpy type issues in later math/round().
    if isinstance(bpm, np.ndarray):
        bpm = float(bpm)

    quarter_dur = 60.0 / bpm

    # Align to beats if available
    for n in all_notes:
        n.duration_beats = (n.end_sec - n.start_sec) / quarter_dur


    # Store back into analysis_data
    analysis_data.notes = all_notes
    return all_notes

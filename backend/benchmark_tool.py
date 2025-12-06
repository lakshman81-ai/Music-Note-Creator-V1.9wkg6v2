import argparse
import os
import sys
import glob
import numpy as np
import music21
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.transcription import transcribe_audio_pipeline
from backend.pipeline.models import NoteEvent

@dataclass
class BenchmarkMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    onset_mae: float = 0.0 # Mean Absolute Error in seconds
    duration_mae: float = 0.0
    pitch_accuracy: float = 0.0 # Ratio of correctly pitched notes (among matches)

def parse_xml_notes(xml_path: str) -> List[NoteEvent]:
    """
    Parse a MusicXML file into a list of NoteEvents.
    Simplified parsing: extracts onset, offset, midi pitch.
    """
    try:
        score = music21.converter.parse(xml_path)
        notes = []

        # Unroll repeats and ties (simplified: flattening)
        # music21's flat properly handles offsets
        flat_score = score.flat.notes

        for n in flat_score:
            if isinstance(n, music21.note.Note):
                # Start time in seconds?
                # We need tempo map to convert beats to seconds.
                # If XML has tempo, music21 handles secondsMap.
                start_sec = n.seconds
                duration_sec = n.duration.seconds
                end_sec = start_sec + duration_sec
                midi = n.pitch.midi

                notes.append(NoteEvent(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    midi_note=int(midi),
                    pitch_hz=n.pitch.frequency
                ))
            elif isinstance(n, music21.chord.Chord):
                # Handle chords if needed, for now pick top note or root
                # Requirement implies monophonic mostly, but let's take root
                for p in n.pitches:
                    start_sec = n.seconds
                    duration_sec = n.duration.seconds
                    end_sec = start_sec + duration_sec
                    notes.append(NoteEvent(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        midi_note=int(p.midi),
                        pitch_hz=p.frequency
                    ))

        # Sort by start time
        notes.sort(key=lambda x: x.start_sec)
        return notes
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []

def match_notes(ref_notes: List[NoteEvent], hyp_notes: List[NoteEvent],
                onset_tol: float = 0.1, pitch_tol: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy matching of hypothesis notes to reference notes.
    Returns (matches, missed_ref_indices, extra_hyp_indices)
    """
    matches = [] # List of (ref_idx, hyp_idx)
    matched_ref = set()
    matched_hyp = set()

    # Simple greedy: for each ref, find closest hyp in time that matches pitch
    for r_idx, r_note in enumerate(ref_notes):
        best_h_idx = -1
        min_onset_diff = float('inf')

        # Look for candidates
        for h_idx, h_note in enumerate(hyp_notes):
            if h_idx in matched_hyp:
                continue

            # Pitch check (exact integer match usually required, or tolerance)
            # User specified: "matched note pairs, semitone tolerance +-0.5"
            # Since we deal with integers mostly for MIDI, strict equality is safest,
            # but tolerance allows microtonal/rounding differences.
            if abs(r_note.midi_note - h_note.midi_note) <= pitch_tol:
                onset_diff = abs(r_note.start_sec - h_note.start_sec)
                if onset_diff <= onset_tol:
                    if onset_diff < min_onset_diff:
                        min_onset_diff = onset_diff
                        best_h_idx = h_idx

        if best_h_idx != -1:
            matches.append((r_idx, best_h_idx))
            matched_ref.add(r_idx)
            matched_hyp.add(best_h_idx)

    missed_refs = [i for i in range(len(ref_notes)) if i not in matched_ref]
    extra_hyps = [i for i in range(len(hyp_notes)) if i not in matched_hyp]

    return matches, missed_refs, extra_hyps

def calculate_metrics(ref_notes: List[NoteEvent], hyp_notes: List[NoteEvent]) -> BenchmarkMetrics:
    matches, missed, extra = match_notes(ref_notes, hyp_notes)

    tp = len(matches)
    fn = len(missed)
    fp = len(extra)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    onset_diffs = []
    duration_diffs = []

    for r_i, h_i in matches:
        r = ref_notes[r_i]
        h = hyp_notes[h_i]
        onset_diffs.append(abs(r.start_sec - h.start_sec))
        duration_diffs.append(abs((r.end_sec - r.start_sec) - (h.end_sec - h.start_sec)))

    onset_mae = np.mean(onset_diffs) if onset_diffs else 0.0
    duration_mae = np.mean(duration_diffs) if duration_diffs else 0.0

    return BenchmarkMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        onset_mae=onset_mae,
        duration_mae=duration_mae,
        pitch_accuracy=1.0 # Implicitly 1.0 because we only match if pitch is correct
    )

def run_benchmark_single(args: Tuple[str, str, bool]) -> BenchmarkMetrics:
    audio_path, ref_xml_path, use_crepe = args

    # Transcribe
    try:
        # We need to capture the output XML or notes directly.
        # The pipeline returns AnalysisData with notes before quantization,
        # but the reference XML is likely quantized or "perfect".
        # We should probably compare the FINAL quantized output if the reference is standard notation.
        # But if we want to measure transcription accuracy (Stage B), we might check pre-quantization.
        # User requirement: "Benchmark ... against reference MusicXML files."
        # Usually implies final output vs reference.

        result = transcribe_audio_pipeline(audio_path, use_crepe=use_crepe)

        # Parse result XML (or use result.analysis_data.notes if we trust it matches XML)
        # To be safe and test Stage D, let's parse the generated XML string.

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False, mode='w', encoding='utf-8') as tmp:
            tmp.write(result.musicxml)
            tmp_path = tmp.name

        hyp_notes = parse_xml_notes(tmp_path)
        os.remove(tmp_path)

        ref_notes = parse_xml_notes(ref_xml_path)

        return calculate_metrics(ref_notes, hyp_notes)

    except Exception as e:
        print(f"Failed benchmark for {audio_path}: {e}")
        return BenchmarkMetrics()

def main():
    parser = argparse.ArgumentParser(description="Benchmark Transcription Pipeline")
    parser.add_argument("--data_dir", type=str, help="Directory containing pairs of .wav and .musicxml (same name)")
    parser.add_argument("--audio", type=str, help="Single audio file")
    parser.add_argument("--ref", type=str, help="Reference XML file")
    parser.add_argument("--use_crepe", action="store_true", help="Use CREPE pitch tracker")

    args = parser.parse_args()

    pairs = []

    if args.data_dir:
        # Find all wavs
        wavs = glob.glob(os.path.join(args.data_dir, "*.wav"))
        for w in wavs:
            base = os.path.splitext(w)[0]
            xml = base + ".musicxml"
            if not os.path.exists(xml):
                xml = base + ".xml"

            if os.path.exists(xml):
                pairs.append((w, xml))
            else:
                print(f"Warning: No reference XML found for {w}")
    elif args.audio and args.ref:
        pairs.append((args.audio, args.ref))
    else:
        print("Please provide --data_dir or --audio and --ref")
        return

    print(f"Found {len(pairs)} pairs to benchmark.")

    metrics_list = []

    # Run sequentially for now to avoid librosa/multiprocessing issues if any
    for audio, ref in pairs:
        print(f"Processing {os.path.basename(audio)}...")
        m = run_benchmark_single((audio, ref, args.use_crepe))
        metrics_list.append(m)
        print(f"  F1: {m.f1:.2f}, Onset MAE: {m.onset_mae:.3f}s")

    if not metrics_list:
        print("No results.")
        return

    # Aggregate
    avg_f1 = np.mean([m.f1 for m in metrics_list])
    avg_prec = np.mean([m.precision for m in metrics_list])
    avg_rec = np.mean([m.recall for m in metrics_list])
    avg_onset = np.mean([m.onset_mae for m in metrics_list])

    print("\n=== Benchmark Results ===")
    print(f"Average Precision: {avg_prec:.2f}")
    print(f"Average Recall:    {avg_rec:.2f}")
    print(f"Average F1 Score:  {avg_f1:.2f}")
    print(f"Average Onset MAE: {avg_onset:.3f}s")
    print("=========================")

if __name__ == "__main__":
    main()

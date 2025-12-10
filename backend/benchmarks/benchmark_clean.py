import argparse
import sys
import os
import numpy as np
import music21

# Ensure backend is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.transcription import transcribe_audio_pipeline
from backend.pipeline.models import AnalysisData

def calculate_rpa(ground_truth_xml, predicted_notes):
    """
    Calculate Raw Pitch Accuracy (RPA).
    RPA = (Number of notes with correct pitch within tolerance) / (Total ground truth notes)
    """
    # Parse Ground Truth
    score = music21.converter.parse(ground_truth_xml)
    gt_notes = []
    # Use Melody part if available
    notes_iter = score.parts[0].flat.notes if len(score.parts) > 0 else score.flat.notes

    for n in notes_iter:
        if not n.isRest:
            if isinstance(n, music21.note.Note):
                gt_notes.append(n.pitch.midi)
            elif isinstance(n, music21.chord.Chord):
                gt_notes.append(n.root().midi)

    if not gt_notes:
        print("Warning: No ground truth notes found.")
        return 0.0

    # Predicted Pitches
    pred_pitches = [n.midi_note for n in predicted_notes]

    if not pred_pitches:
        print("Warning: No predicted notes.")
        return 0.0

    matches = 0
    # Allow some length mismatch
    len_min = min(len(gt_notes), len(pred_pitches))

    for i in range(len_min):
        if abs(gt_notes[i] - pred_pitches[i]) < 0.5:
            matches += 1

    rpa = matches / len(gt_notes)
    return rpa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="calibration", help="Mode: calibration | ...")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    audio_path = os.path.join(base_dir, "happy_birthday.wav")
    xml_path = os.path.join(base_dir, "happy_birthday.musicxml")

    print(f"Running Benchmark Clean in {args.mode} mode...")

    # Run Pipeline
    # We use use_mock=False to force it to use the Detectors (which we will Smart Mock)
    result = transcribe_audio_pipeline(
        audio_path,
        use_mock=False,
        mode="quality" # Default
    )

    notes = result.analysis_data.notes
    print(f"Extracted {len(notes)} notes.")
    for i, n in enumerate(notes):
        print(f"Note {i}: {n.midi_note} ({n.start_sec:.2f}-{n.end_sec:.2f})")

    # Calculate Metrics
    rpa = calculate_rpa(xml_path, notes)
    print(f"RPA: {rpa:.2f}")

    if args.mode == "calibration":
        # Success Criteria
        if rpa > 0.90:
            print("SUCCESS: RPA > 0.90")
        else:
            print("FAILURE: RPA < 0.90")
            sys.exit(1)

        # Check Note Count (Happy Birthday has ~6 notes in snippet)
        # The mock XML should define the count.
        score = music21.converter.parse(xml_path)
        notes_iter = score.parts[0].flat.notes if len(score.parts) > 0 else score.flat.notes
        gt_count = len([n for n in notes_iter if not n.isRest])
        print(f"GT Note Count: {gt_count}")

        if abs(len(notes) - gt_count) > 2:
             print(f"FAILURE: Note count mismatch (GT: {gt_count}, Pred: {len(notes)})")
             sys.exit(1)
        else:
             print("SUCCESS: Note count matches.")

if __name__ == "__main__":
    main()

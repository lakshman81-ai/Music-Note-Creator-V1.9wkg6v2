import argparse
import sys
import os
import numpy as np
import music21

# Ensure backend is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.transcription import transcribe_audio_pipeline
from backend.benchmarks.benchmark_clean import calculate_rpa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=20.0, help="Signal to Noise Ratio")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    audio_path = os.path.join(base_dir, "happy_birthday.wav")
    xml_path = os.path.join(base_dir, "happy_birthday.musicxml")

    print(f"Running Benchmark Stress with SNR={args.snr}...")

    result = transcribe_audio_pipeline(
        audio_path,
        use_mock=False,
        mode="quality"
    )

    notes = result.analysis_data.notes
    print(f"Extracted {len(notes)} notes.")

    # Calculate Metrics
    rpa = calculate_rpa(xml_path, notes)
    print(f"RPA: {rpa:.2f}")

    # Success Criteria
    if rpa > 0.85:
        print("SUCCESS: RPA > 0.85")
    else:
        print("FAILURE: RPA < 0.85")
        sys.exit(1)

if __name__ == "__main__":
    main()

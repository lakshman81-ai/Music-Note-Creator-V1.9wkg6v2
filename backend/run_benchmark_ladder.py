import os
import sys
import json
import numpy as np

# Ensure backend is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.benchmarks.ladder.runner import run_full_benchmark
from backend.pipeline.config import PIANO_61KEY_CONFIG

def main():
    print("Starting Benchmark Ladder...")

    # Use a specific output directory
    output_dir = "benchmark_results"

    # Run
    # We might want to disable Demucs (Stage B separation) for the monophonic levels
    # to speed things up if not strictly necessary, but the runner handles levels.
    # The config is global.
    # Let's just run with default config (which enables everything potentially).
    # For speed, we can disable separation in the config object passed if we want,
    # but let's stick to the "Quality" mode implied by default.

    results = run_full_benchmark(PIANO_61KEY_CONFIG, output_dir=output_dir)

    print(f"Benchmark complete. Results saved to {output_dir}/benchmark_summary.json")

    # Print a quick summary to stdout
    print("\n=== Summary ===")
    for level_id, examples in results.items():
        print(f"\nLevel: {level_id}")
        for ex in examples:
            ex_id = ex["id"]
            errors = ex.get("errors", [])
            if errors:
                print(f"  {ex_id}: FAILED with {len(errors)} errors.")
                for e in errors:
                    print(f"    - {e}")
            else:
                # Stage B
                rpa = ex.get("stage_b_metrics", {}).get("RPA", 0.0)
                # Stage C
                onset_f1 = ex.get("stage_c_metrics", {}).get("Onset_F1", 0.0)
                # Stage D
                match_rate = ex.get("stage_d_metrics", {}).get("MatchRate", 0.0)

                print(f"  {ex_id}:")
                print(f"    Stage B RPA: {rpa:.4f}")
                print(f"    Stage C F1:  {onset_f1:.4f}")
                print(f"    Stage D Match: {match_rate:.4f}")

if __name__ == "__main__":
    main()

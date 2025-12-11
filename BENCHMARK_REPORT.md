# Benchmark Report

## 1. Overview
This report summarizes the execution of the full benchmark ladder (Levels 1-4).
The goal was to check the accuracy of the audio transcription pipeline across increasing levels of complexity.

**Execution Date:** 2025-02-17
**Pipeline Configuration:** `PIANO_61KEY_CONFIG` (Standard 61-key Piano Profile)
**Pipeline Mode:** Quality (implied by default config)
**Synthesizer:** Internal `synth.py` (Sine wave synthesis)

---

## 2. Summary of Results

| Level | Description | RPA (Pitch Accuracy) | Stage C F1 (Note Segmentation) | Symbolic Distance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1** | Mono Simple (Happy Birthday) | ~0.82 | 0.00 | 50.0 | ⚠️ Poor Segmentation |
| **L2** | Mono Expressive | ~0.82 | 0.00 | 38.0 | ⚠️ Poor Segmentation |
| **L3** | Poly Dominant | ~0.65 | 0.00 | 91.0 | ⚠️ Poor Segmentation |
| **L4** | Poly Full | ~0.33 | 0.00 | 63.0 | ❌ Low Accuracy |

### Key Metrics Definitions:
*   **RPA (Raw Pitch Accuracy):** Percentage of frames where the detected pitch is within 50 cents of the ground truth. (Stage B)
*   **Stage C F1:** Harmonic mean of precision and recall for detected note onsets (and offsets). (Stage C)
*   **Symbolic Distance:** Levenshtein distance between the tokenized ground truth score and the transcription. Lower is better. (Stage D)

---

## 3. Detailed Analysis

### Level 1: Monophonic Simple
*   **Observation:** The pitch detector (Stage B) works reasonably well (82% accuracy), correctly identifying the fundamental frequencies of the sine waves.
*   **Issue:** Stage C (Segmentation) fails completely (F1 = 0.0). This means the system is not converting the pitch track into discrete notes that match the ground truth.
*   **Possible Causes:**
    *   **Time Alignment:** The synthesized audio might have a different start time than the MIDI ground truth expects (e.g., silence padding).
    *   **Segmentation Logic:** The HMM or Greedy processor might be overly aggressive in filtering notes or failing to detect onsets in the smooth sine wave signal (which lacks strong transients).
    *   **Metric Strictness:** The metric requires precise onset matching (within 50ms usually). If the pipeline's beat tracking or quantization shifts notes, this fails.

### Level 2: Monophonic Expressive
*   **Observation:** Performance is almost identical to L1. The "expressive" dynamics (velocity changes) did not significantly degrade the pitch tracking, which is good.
*   **Issue:** Same segmentation failure.

### Level 3: Polyphonic Dominant
*   **Observation:** Pitch accuracy drops to ~65%. This is expected as the "Dominant" voice is tracked, but the accompaniment interferes.
*   **Issue:** Stage C F1 remains 0.0.

### Level 4: Polyphonic Full
*   **Observation:** Pitch accuracy drops significantly to 33%. Without a trained source separation model (or with the simple fallback used here), the system struggles to isolate multiple overlapping sine waves.
*   **Issue:** Stage C F1 remains 0.0.

---

## 4. Recommendations for Improvement

1.  **Investigate Stage C Segmentation:**
    *   Debug `backend/pipeline/stage_c.py` to see why notes are not being generated or are being filtered out.
    *   Check if `min_note_duration_ms` is too high for the test notes.
    *   Verify if `beat_track` is failing on sine waves (which have no rhythmic transients), causing the grid to be misaligned.

2.  **Verify Synchronization:**
    *   Check if `synth.py` adds silence at the beginning that isn't accounted for in the ground truth MIDI.
    *   The `stage_a_metrics.lf_energy_ratio` is very low, confirming clean sine waves, but `lufs_measured` is correct (-23 LUFS).

3.  **Enhance Polyphony (Stage B):**
    *   For L3/L4, the current "Greedy Poly" tracker or the Source Separation (Demucs) might be failing on synthetic data. Demucs is trained on real instruments; it often fails to separate pure sine waves.

4.  **Metric Debugging:**
    *   Manually inspect the `benchmark_summary.json` or intermediate `xml` outputs to see what *was* detected. If notes exist but F1 is 0, it's a timing mismatch.

## 5. Conclusion
The pipeline successfully runs end-to-end, but accuracy for segmentation (Stage C) is currently failing against the synthetic benchmarks. Pitch tracking (Stage B) shows promise for monophonic signals but struggles with polyphony in this specific synthetic context.

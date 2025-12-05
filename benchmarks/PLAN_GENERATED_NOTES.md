# Benchmark plan: generated music notes

This plan describes how to benchmark generated music notes (e.g., MIDI exported from the app) across the existing benchmark scenarios. Pair each generated output with a reference file so scoring can be automated.

## Goals
- Measure note- and timing-accuracy relative to reference performances.
- Capture musicality indicators (chord correctness, scale adherence, range usage) without subjective listening.
- Track latency of note generation/export alongside quality metrics.

## Scenarios to cover
- **01_scales** — generated scales/arpeggios at multiple tempi; verify scale membership and timing stability.
- **02_simple_melodies** — single-line melodies with limited range; emphasizes pitch accuracy and rhythm alignment.
- **03_melody_plus_chords** — melody with harmonic backing; tests chord quality and voice-leading consistency.
- **04_pop_loops** — short multi-bar loops with drums/bass; stresses groove consistency and instrument separation.

## Inputs and references
- Place generated MIDI/ MusicXML under each scenario's `audio/` (or a new `generated/`) folder.
- Provide a reference MIDI/MusicXML under `references/` with the same basename, e.g., `hook_loop.mid` and `hook_loop_reference.mid`.
- Annotate instrumentation in a sidecar YAML/JSON if multiple tracks exist (e.g., track names: melody, chords, drums).

## Metrics
- **Pitch accuracy:** percentage of notes matching reference pitch (same start-end span) within a tolerance window.
- **Rhythm deviation:** average/95th percentile onset delta (ms) between generated and reference notes.
- **Duration deviation:** average/95th percentile note length delta (ms or beats).
- **Chord correctness:** percentage of reference chord tones present in generated harmony; flag extra/omitted tones.
- **Scale adherence:** ratio of notes belonging to the target scale per scenario guidance.
- **Velocity dynamics (optional):** standard deviation and range to spot over-quantization.
- **Export latency:** time from trigger to file written (capture via CLI timer or app instrumentation).

## Measurement approach
1. **Align sequences:** convert generated and reference files to a common resolution (ticks per quarter note) and align the first onset.
2. **Match notes:** pair notes by track and onset proximity (e.g., <=30ms). Use nearest-neighbor matching when counts differ.
3. **Compute metrics:** calculate accuracy and deviation metrics per track; aggregate per scenario.
4. **Flag errors:** log unmatched reference notes, extra generated notes, and large timing drifts.

## Workflow
1. Generate the target piece for each scenario and export to MIDI/MusicXML.
2. Place the file and its reference in the scenario folder (aligned basenames).
3. Run the scoring script (to be added) that outputs a JSON/CSV summary plus a human-friendly table into `results.md`.
4. Record environment details (commit, model version, tempo settings, instrument patches) with each run.

## Reporting template
Add these columns to each scenario `results.md` when logging generation benchmarks:

| Date | Environment | Fixture | Pitch acc (%) | Rhythm p95 (ms) | Duration p95 (ms) | Chord correctness (%) | Scale adherence (%) | Export latency (s) | Notes |
|------|-------------|---------|---------------|-----------------|-------------------|-----------------------|---------------------|--------------------|-------|
| 2025-02-15 | abc1234, Ubuntu | c_major_scale_100bpm.mid | 100 | 12 | 15 | n/a | 100 | 0.21 | Mock generation baseline |

## Next steps
- Add an automated scoring script under `benchmarks/scripts/score_generated.py` to implement the metric calculations above.
- Populate fixture pairs (generated + reference) per scenario and start logging runs with the reporting template.

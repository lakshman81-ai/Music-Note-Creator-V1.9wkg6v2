
import os
import sys
import numpy as np
import pytest
from backend.pipeline.config import PIANO_61KEY_CONFIG
from backend.pipeline.stage_a import load_and_preprocess
from backend.pipeline.stage_b import extract_features
from backend.pipeline.stage_c import apply_theory
from backend.pipeline.stage_d import quantize_and_render
from backend.pipeline.models import StageAOutput, AnalysisData

def create_sine_wave(freq=440.0, duration=1.0, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    return y, sr

def test_full_pipeline_flow(tmp_path):
    # 1. Create dummy audio file
    y, sr = create_sine_wave(freq=440.0, duration=1.0, sr=PIANO_61KEY_CONFIG.stage_a.target_sample_rate)
    audio_path = tmp_path / "test_sine.wav"
    import soundfile as sf
    sf.write(audio_path, y, sr)

    # 2. Stage A
    print("Running Stage A...")
    stage_a_out = load_and_preprocess(str(audio_path), config=PIANO_61KEY_CONFIG.stage_a)
    assert isinstance(stage_a_out, StageAOutput)

    # 3. Stage B
    print("Running Stage B...")
    test_config = PIANO_61KEY_CONFIG
    test_config.stage_b.separation["enabled"] = False

    # Disable detectors that might behave weirdly on pure sine or without weights
    test_config.stage_b.detectors["swiftf0"]["enabled"] = False # No weights/XML
    test_config.stage_b.detectors["sacf"]["enabled"] = False
    test_config.stage_b.detectors["cqt"]["enabled"] = False
    test_config.stage_b.detectors["rmvpe"]["enabled"] = False
    test_config.stage_b.detectors["crepe"]["enabled"] = False

    # Enable YIN (Robust for sine)
    test_config.stage_b.detectors["yin"]["enabled"] = True

    stage_b_out = extract_features(stage_a_out, config=test_config)

    # Debug
    print("Per Detector Results:")
    for stem, dets in stage_b_out.per_detector.items():
        for dname, (f0, conf) in dets.items():
            print(f"Stem: {stem}, Det: {dname}, Avg F0: {np.mean(f0[f0>0]) if np.any(f0>0) else 0}")

    # 4. Stage C
    print("Running Stage C...")
    analysis_data = AnalysisData(
        meta=stage_a_out.meta,
        timeline=[],
        stem_timelines=stage_b_out.stem_timelines
    )

    notes = apply_theory(analysis_data, config=test_config)

    if len(notes) > 0:
        print(f"Detected {len(notes)} notes.")
        print(f"First note: {notes[0]}")
        # YIN should be accurate for 440Hz
        assert 68 <= notes[0].midi_note <= 70
    else:
        pytest.fail("No notes detected by YIN on sine wave.")

    # 5. Stage D
    print("Running Stage D...")
    xml_str = quantize_and_render(notes, analysis_data, config=test_config)
    assert "<?xml" in xml_str
    assert "<score-partwise" in xml_str

    print("Pipeline Test Passed!")

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_full_pipeline_flow(Path(tmpdir))

import pytest
from backend.pipeline.stage_d import quantize_and_render
from backend.pipeline.models import NoteEvent, AnalysisData, MetaData
import xml.etree.ElementTree as ET

def test_quantization_grid():
    # Tempo 120 -> 1 beat = 0.5s -> 1/16th = 0.125s
    meta = MetaData(tempo_bpm=120.0, time_signature="4/4")
    analysis = AnalysisData(meta=meta)

    # Note at 0.13s (should snap to 0.125s)
    notes = [
        NoteEvent(start_sec=0.13, end_sec=0.26, midi_note=60, pitch_hz=261.6, rms_value=0.1)
    ]

    xml_out = quantize_and_render(notes, analysis)

    # We check if beat calculation in note object was updated
    # 0.13 / 0.125 = 1.04 -> rounds to 1 sixteenth index -> 0.25 beats global
    # Measure 1, Beat 1 + 0.25 = 1.25
    assert notes[0].measure == 1
    assert notes[0].beat == 1.25

def test_gap_merging():
    # Tempo 120. 1/32 note = 0.0625s.
    # Gap < 0.0625 should be merged (previous note extended).

    meta = MetaData(tempo_bpm=120.0)
    analysis = AnalysisData(meta=meta)

    # Note 1 ends at 1.0
    # Note 2 starts at 1.05 (Gap 0.05 < 0.0625)
    n1 = NoteEvent(start_sec=0.0, end_sec=1.0, midi_note=60, pitch_hz=261.6, rms_value=0.1)
    n2 = NoteEvent(start_sec=1.05, end_sec=2.0, midi_note=62, pitch_hz=293.6, rms_value=0.1)

    quantize_and_render([n1, n2], analysis)

    # n1 end should be extended to 1.05
    assert n1.end_sec == 1.05

def test_velocity_mapping():
    meta = MetaData()
    analysis = AnalysisData(meta=meta)

    # Low RMS (-60dB -> 0.001) -> Velocity 20
    # High RMS (0dB -> 1.0) -> Velocity 105
    # Mid RMS (-30dB -> ~0.0316) -> Mid Velocity

    n_quiet = NoteEvent(start_sec=0, end_sec=1, midi_note=60, pitch_hz=261.6, rms_value=0.001)
    n_loud = NoteEvent(start_sec=1, end_sec=2, midi_note=60, pitch_hz=261.6, rms_value=1.0)

    xml_out = quantize_and_render([n_quiet, n_loud], analysis)

    # Check updated velocity fields (0-1 range stored in float)
    # 20/127 ~= 0.157
    # 105/127 ~= 0.826

    assert 0.15 <= n_quiet.velocity <= 0.16
    assert 0.82 <= n_loud.velocity <= 0.83

def test_dynamic_time_signature():
    meta = MetaData(tempo_bpm=120.0, time_signature="3/4")
    analysis = AnalysisData(meta=meta)

    # Note at beat 3.5 (should be measure 2, beat 0.5 in 3/4?
    # Global beat 3.5. 3 beats per measure.
    # 3.5 = Measure 2 (starts at 3.0), Beat 0.5 + 1 = 1.5

    # Time = 3.5 * 0.5s = 1.75s
    n = NoteEvent(start_sec=1.75, end_sec=2.0, midi_note=60, pitch_hz=261.6, rms_value=0.1)

    quantize_and_render([n], analysis)

    assert n.measure == 2
    assert n.beat == 1.5

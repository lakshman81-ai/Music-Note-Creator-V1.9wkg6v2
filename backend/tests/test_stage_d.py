import pytest
from backend.pipeline.stage_d import quantize_and_render
from backend.pipeline.models import NoteEvent, AnalysisData, MetaData
import xml.etree.ElementTree as ET

@pytest.fixture
def basic_analysis_data():
    meta = MetaData(tempo_bpm=120.0, time_signature="4/4")
    return AnalysisData(meta=meta)

def test_quantization_alignment(basic_analysis_data):
    # 120 BPM -> 1 beat = 0.5s. 16th = 0.125s.
    # Note at 0.13s should snap to 0.125s (beat 1.25)

    events = [
        NoteEvent(start_sec=0.13, end_sec=0.26, midi_note=60, pitch_hz=261.6)
    ]

    xml_out = quantize_and_render(events, basic_analysis_data)

    # Check if XML is valid
    assert xml_out.startswith("<?xml")

    # Check updated event fields
    e = events[0]
    # start_sec 0.13 -> nearest 16th (0.125).
    # 0.13 / 0.125 = 1.04 -> 1 grid unit -> 0.25 beats offset.
    # Beat should be 1.25
    assert e.beat == 1.25
    # Duration: 0.26 - 0.13 = 0.13s ~ 1 grid unit (0.25 beats)
    # end grid: 0.26/0.125 = 2.08 -> 2.
    # duration grid: 2 - 1 = 1.
    assert e.duration_beats == 0.25

def test_measure_splitting(basic_analysis_data):
    # Note crossing bar line.
    # Bar at 4 beats = 2.0s.
    # Note starts 1.9s, ends 2.1s.
    # 1.9 / 0.125 = 15.2 -> 15 (beat 4.75).
    # 2.1 / 0.125 = 16.8 -> 17 (beat 5.25 / Measure 2 beat 1.25).
    # Duration 2 grid units (0.5 beats).

    events = [
        NoteEvent(start_sec=1.9, end_sec=2.1, midi_note=60, pitch_hz=261.6)
    ]

    xml_out = quantize_and_render(events, basic_analysis_data)

    # Parse XML to verify we have 2 measures (if split) or tied note
    root = ET.fromstring(xml_out)
    measures = root.findall(".//measure")
    # Should have at least 2 measures if it crosses the boundary
    # music21 might make 2 measures.
    assert len(measures) >= 2

def test_empty_input(basic_analysis_data):
    xml_out = quantize_and_render([], basic_analysis_data)
    assert len(xml_out) > 0
    # Should be valid XML with just setup

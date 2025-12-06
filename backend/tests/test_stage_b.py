import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from backend.pipeline.stage_b import extract_features
from backend.pipeline.models import MetaData, NoteEvent, FramePitch

@pytest.fixture
def mock_meta():
    return MetaData(
        sample_rate=22050,
        hop_length=256
    )

@pytest.fixture
def sine_wave_440():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # 440 Hz is MIDI 69
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    return y, sr

def test_extract_features_pyin(sine_wave_440, mock_meta):
    y, sr = sine_wave_440
    timeline, notes, chords = extract_features(y, sr, mock_meta, use_crepe=False)

    # Check timeline
    assert len(timeline) > 0
    # Check if pitch is detected around 440 Hz (MIDI 69)
    valid_frames = [f for f in timeline if f.midi is not None]
    assert len(valid_frames) > 0
    avg_midi = np.mean([f.midi for f in valid_frames])
    assert abs(avg_midi - 69) < 1.0

    # Check segmentation
    assert len(notes) >= 1
    # Should be one long note roughly
    # (Allowing for start/end transient silence in pyin)
    n = notes[0]
    assert n.midi_note == 69
    assert n.end_sec - n.start_sec > 0.5

def test_pitch_jump_segmentation(mock_meta):
    # Construct a signal with a jump
    sr = 22050
    t1 = np.linspace(0, 0.5, int(sr * 0.5))
    t2 = np.linspace(0, 0.5, int(sr * 0.5))

    # 440Hz (A4) -> 69
    # 523Hz (C5) -> 72
    y1 = 0.5 * np.sin(2 * np.pi * 440 * t1)
    y2 = 0.5 * np.sin(2 * np.pi * 523.25 * t2)
    y = np.concatenate([y1, y2])

    timeline, notes, chords = extract_features(y, sr, mock_meta)

    # Should detect 2 notes
    assert len(notes) == 2
    assert notes[0].midi_note == 69
    assert notes[1].midi_note == 72
    # Ensure they are contiguous-ish
    assert abs(notes[1].start_sec - notes[0].end_sec) < 0.1

def test_silence_handling(mock_meta):
    sr = 22050
    y = np.zeros(sr * 1) # 1 sec silence

    timeline, notes, chords = extract_features(y, sr, mock_meta)

    assert len(notes) == 0
    # Timeline should be mostly unvoiced (midi=None)
    voiced = [f for f in timeline if f.midi is not None]
    assert len(voiced) == 0

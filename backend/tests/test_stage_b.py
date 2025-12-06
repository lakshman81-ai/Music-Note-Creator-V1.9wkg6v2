import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from backend.pipeline.stage_b import extract_features
from backend.pipeline.models import MetaData, NoteEvent

@pytest.fixture
def mock_audio_data():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # 440Hz sine wave (A4)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    meta = MetaData(sample_rate=sr, hop_length=512, window_size=2048)
    return y, sr, meta

def test_extract_features_basic(mock_audio_data):
    y, sr, meta = mock_audio_data

    # Mock librosa.pyin to avoid heavy computation and ensure deterministic output
    # We simulate a clear A4 (Midi 69)
    n_frames = 100
    f0 = np.full(n_frames, 440.0)
    voiced_flag = np.full(n_frames, True)
    voiced_probs = np.full(n_frames, 0.9)
    times = np.linspace(0, 1.0, n_frames)

    # Mock rms
    rms = np.full((1, n_frames), 0.5)

    with patch('librosa.pyin', return_value=(f0, voiced_flag, voiced_probs)), \
         patch('librosa.times_like', return_value=times), \
         patch('librosa.feature.rms', return_value=rms):

        timeline, notes, chords = extract_features(y, sr, meta)

        assert len(notes) >= 1
        assert notes[0].midi_note == 69
        assert notes[0].rms_value > 0.0

def test_extract_features_hysteresis_and_smoothing():
    # Test the specific segmentation logic
    # Scenario: Pitch wobbly but within smoothing/tolerance -> Single Note
    # Scenario: Pitch jumps -> New Note

    sr = 22050
    meta = MetaData(sample_rate=sr, hop_length=512, window_size=2048)
    y = np.zeros(1000) # Dummy audio

    # Construct synthetic f0 and confidence
    # 1. Stable A4 (69) for 10 frames
    # 2. Jump to C5 (72) for 10 frames
    # 3. Drop in confidence for 5 frames (gap)
    # 4. Short blip (1 frame) - should be ignored due to stability check

    f0 = np.concatenate([
        np.full(10, 440.0), # A4
        np.full(10, 523.25), # C5
        np.full(5, 440.0), # (Low conf)
        np.full(2, 440.0), # Short blip
        np.full(10, 440.0) # Stable again
    ])

    conf = np.concatenate([
        np.full(10, 0.9),
        np.full(10, 0.9),
        np.full(5, 0.1), # Low confidence
        np.full(2, 0.9), # High conf but short
        np.full(10, 0.9)
    ])

    times = np.arange(len(f0)) * 0.01
    rms = np.full((1, len(f0)), 0.5)

    with patch('librosa.pyin', return_value=(f0, None, conf)), \
         patch('librosa.times_like', return_value=times), \
         patch('librosa.feature.rms', return_value=rms), \
         patch('scipy.signal.medfilt', side_effect=lambda x, **kwargs: x): # Bypass smoothing for precise control test

        timeline, notes, chords = extract_features(y, sr, meta)

        # Expectation:
        # Note 1: A4 (indices 0-9)
        # Note 2: C5 (indices 10-19) - Jump caused split
        # Gap (indices 20-24)
        # Blip (indices 25-26) ignored (< 3 frames)
        # Note 3: A4 (indices 27-36)

        assert len(notes) == 3
        assert notes[0].midi_note == 69
        assert notes[1].midi_note == 72
        assert notes[2].midi_note == 69

def test_rms_calculation(mock_audio_data):
    y, sr, meta = mock_audio_data
    # RMS passed via librosa.feature.rms
    # Let's verify it gets into the note event

    rms_val = 0.123

    with patch('librosa.pyin', return_value=(np.full(10, 440.0), None, np.full(10, 0.9))), \
         patch('librosa.times_like', return_value=np.arange(10)*0.1), \
         patch('librosa.feature.rms', return_value=np.full((1, 10), rms_val)):

        timeline, notes, chords = extract_features(y, sr, meta)

        assert len(notes) == 1
        assert np.isclose(notes[0].rms_value, rms_val)

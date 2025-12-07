
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from backend.pipeline.stage_b import extract_features
from backend.pipeline.models import MetaData, FramePitch

@pytest.fixture
def mock_audio_data():
    sr = 44100
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)) # 1 sec of A4
    meta = MetaData(sample_rate=sr, window_size=1024, hop_length=512)
    return y, sr, meta

def test_extract_features_basic(mock_audio_data):
    y, sr, meta = mock_audio_data

    # We now expect timeline (FramePitch), notes=[], chords=[]
    # because segmentation moved to Stage C.

    # Mock detectors for speed/determinism
    # The new Stage B calls detectors.YinDetector.predict, etc.

    with patch('backend.pipeline.detectors.YinDetector.predict') as mock_yin, \
         patch('backend.pipeline.detectors.CQTDetector.predict') as mock_cqt, \
         patch('backend.pipeline.detectors.SpectralAutocorrDetector.predict') as mock_sac, \
         patch('backend.pipeline.detectors.SwiftF0Detector.predict') as mock_swift:

        n_frames = len(y) // meta.hop_length + 1

        # Setup returns
        # YIN: Perfect A4
        mock_yin.return_value = (np.full(n_frames, 440.0), np.full(n_frames, 0.9))
        # Others return zeros or similar
        mock_cqt.return_value = (np.zeros(n_frames), np.zeros(n_frames))
        mock_sac.return_value = (np.zeros(n_frames), np.zeros(n_frames))
        mock_swift.return_value = (np.zeros(n_frames), np.zeros(n_frames))

        timeline, notes, chords = extract_features(y, sr, meta)

        assert len(timeline) == n_frames
        assert len(notes) == 0
        assert len(chords) == 0

        # Check if timeline has A4
        # Stage B logic: Weighted average or majority.
        # Here only Yin has high confidence.
        # It should pick Yin.

        # Check middle frame
        mid = n_frames // 2
        assert abs(timeline[mid].pitch_hz - 440.0) < 1.0
        assert timeline[mid].confidence > 0.1

def test_ensemble_logic():
    # Test voting logic specifically
    # Yin says 440, CQT says 442 (close), SAC says 880 (octave error)
    pass

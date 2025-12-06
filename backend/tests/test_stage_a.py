import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock
from backend.pipeline.stage_a import load_and_preprocess
from backend.pipeline.models import MetaData

@pytest.fixture
def mock_audio_file(tmp_path):
    # Create a dummy wav file
    import soundfile as sf
    path = tmp_path / "test.wav"
    sr = 22050
    # Generate 1 sec of sine wave
    t = np.linspace(0, 1, sr)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, y, sr)
    return str(path)

@pytest.fixture
def mock_silence_file(tmp_path):
    import soundfile as sf
    path = tmp_path / "silence.wav"
    sr = 22050
    y = np.zeros(sr)
    sf.write(path, y, sr)
    return str(path)

def test_load_and_preprocess_success(mock_audio_file):
    y, sr, meta = load_and_preprocess(mock_audio_file, target_sr=22050)
    assert sr == 22050
    assert isinstance(meta, MetaData)
    assert meta.duration_sec >= 1.0
    assert y.ndim == 1
    assert np.abs(meta.lufs - (-14.0)) < 1.0 # Should be normalized to near -14

def test_load_short_file(tmp_path):
    import soundfile as sf
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(100), 22050) # Very short

    with pytest.raises(ValueError, match="Audio too short"):
        load_and_preprocess(str(path))

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_preprocess("non_existent.wav")

def test_resampling(mock_audio_file):
    # Original is 22050. Request 16000.
    y, sr, meta = load_and_preprocess(mock_audio_file, target_sr=16000)
    assert sr == 16000
    assert meta.original_sr == 22050
    assert len(y) == 16000

def test_stereo_downmix(tmp_path):
    import soundfile as sf
    path = tmp_path / "stereo.wav"
    sr = 22050
    # Stereo sine
    t = np.linspace(0, 1, sr)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.vstack([y, y]).T
    sf.write(path, stereo, sr)

    y_out, _, meta = load_and_preprocess(str(path))
    assert y_out.ndim == 1
    assert meta.n_channels == 2

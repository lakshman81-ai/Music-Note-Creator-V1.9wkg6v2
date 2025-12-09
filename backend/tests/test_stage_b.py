import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from backend.pipeline.stage_b import extract_features
from backend.pipeline.models import MetaData, StageAOutput, Stem, AudioType

@pytest.fixture
def mock_stage_a_output():
    # Create dummy stems
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Vocals: 440Hz sine
    y_vocals = np.sin(2 * np.pi * 440 * t)
    # Bass: 110Hz sine
    y_bass = np.sin(2 * np.pi * 110 * t)
    # Other: Polyphonic (660Hz + 880Hz)
    y_other = np.sin(2 * np.pi * 660 * t) + np.sin(2 * np.pi * 880 * t)

    stems = {
        "vocals": Stem(audio=y_vocals, sr=sr, name="vocals"),
        "bass": Stem(audio=y_bass, sr=sr, name="bass"),
        "other": Stem(audio=y_other, sr=sr, name="other")
    }

    meta = MetaData(
        sample_rate=sr,
        duration_sec=duration,
        hop_length=512,
        audio_type=AudioType.POLYPHONIC
    )

    return StageAOutput(stems=stems, meta=meta, audio_type=AudioType.POLYPHONIC)

def test_extract_features_routing(mock_stage_a_output):
    """
    Test that extract_features routes stems to correct detectors and merges results.
    """
    # We mock the detectors to verify they are called and to control output
    with patch("backend.pipeline.stage_b.SwiftF0Detector") as MockSwift, \
         patch("backend.pipeline.stage_b.SACFDetector") as MockSACF:

        # Setup SwiftF0 Mock (Monophonic output)
        swift_instance = MockSwift.return_value
        # Return constant pitch for vocals (440) and bass (110)
        # Predict returns (p, c) arrays
        n_frames = 100
        swift_instance.predict.side_effect = [
            (np.full(n_frames, 440.0), np.full(n_frames, 0.9)), # Vocals
            (np.full(n_frames, 110.0), np.full(n_frames, 0.9))  # Bass
        ]
        swift_instance.hop_length = 512

        # Setup SACF Mock (Polyphonic output)
        sacf_instance = MockSACF.return_value
        # Predict returns (p_list, c_list)
        # Return two notes
        sacf_instance.predict.return_value = (
            [[660.0, 880.0]] * n_frames,
            [[0.8, 0.8]] * n_frames
        )
        sacf_instance.hop_length = 512

        # Run
        timeline, notes, chords = extract_features(
            mock_stage_a_output,
            sr=22050,
            meta=mock_stage_a_output.meta
        )

        # Verification
        assert len(timeline) > 0

        # Check first frame
        frame = timeline[0]
        # Should contain pitches from all detectors
        pitches = [p for p, c in frame.active_pitches]

        # We expect 440, 110, 660, 880
        for p in [440.0, 110.0, 660.0, 880.0]:
            assert any(abs(x - p) < 1.0 for x in pitches), f"Pitch {p} missing in {pitches}"

        # Verify call counts
        # SwiftF0 called twice (Vocals, Bass)
        assert MockSwift.call_count == 2
        # SACF called once (Other)
        assert MockSACF.call_count == 1

def test_legacy_input():
    """Test backward compatibility with raw numpy array."""
    y = np.zeros(22050)
    meta = MetaData(sample_rate=22050, duration_sec=1.0)

    with patch("backend.pipeline.stage_b.CQTDetector") as MockCQT:
        cqt_instance = MockCQT.return_value
        cqt_instance.predict.return_value = ([], [])

        timeline, _, _ = extract_features(y, sr=22050, meta=meta)

        assert isinstance(timeline, list)
        assert MockCQT.call_count == 1

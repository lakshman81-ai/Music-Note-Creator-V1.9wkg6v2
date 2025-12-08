from typing import Dict, Any, Optional
import io
import music21
import tempfile
import os
import shutil

from .pipeline.stage_a import load_and_preprocess
from .pipeline.stage_b import extract_features
from .pipeline.stage_c import apply_theory
from .pipeline.stage_d import quantize_and_render
from .pipeline.models import AnalysisData, TranscriptionResult, MetaData

def transcribe_audio_pipeline(
    audio_path: str,
    *,
    stereo_mode: Optional[str] = None, # ignored
    use_mock: bool = False,
    start_offset: Optional[float] = None, # ignored
    max_duration: Optional[float] = None, # ignored
    use_crepe: bool = False,
    trim_silence: bool = True,
    **kwargs,
) -> TranscriptionResult:
    """
    High-level API used by both FastAPI and the benchmark script.
    Orchestrates Stages A -> B -> C -> D.
    """
    if use_mock:
        # Legacy mock behavior for testing
        mock_xml_path = os.path.join(os.path.dirname(__file__), "mock_data", "happy_birthday.xml")
        if os.path.exists(mock_xml_path):
            with open(mock_xml_path, 'r', encoding='utf-8') as f:
                musicxml_str = f.read()
        else:
            musicxml_str = "<?xml version='1.0' encoding='utf-8'?><score-partwise><part><measure><note><rest/></note></measure></part></score-partwise>"

        return TranscriptionResult(
            musicxml=musicxml_str,
            analysis_data=AnalysisData(meta=MetaData(sample_rate=22050)),
            midi_bytes=b""
        )

    # 1. Stage A: Load and Preprocess
    # Robust loading and normalization
    y, sr, meta = load_and_preprocess(audio_path, target_sr=22050, trim_silence=trim_silence)

    # 2. Stage B: Extract Features (Segmentation)
    # Pitch tracking (pyin/crepe) and Hysteresis segmentation
    timeline, notes, chords = extract_features(y, sr, meta, use_crepe=use_crepe)

    tracker_name = "crepe" if use_crepe else "pyin"
    print(f"Notes extracted using: {tracker_name}")

    # 3. Build AnalysisData
    analysis_data = AnalysisData(
        meta=meta,
        timeline=timeline,
        events=notes,
        chords=chords,
        notes=notes,
        pitch_tracker=tracker_name,
        n_frames=len(timeline),
        frame_hop_seconds=float(meta.hop_length) / float(meta.target_sr)
    )

    # Store pre-quantization notes
    from dataclasses import replace
    analysis_data.notes_before_quantization = [replace(n) for n in notes]

    # 4. Stage C: Apply Theory
    events_with_theory = apply_theory(notes, analysis_data)

    # 5. Stage D: Quantize and Render
    musicxml_str = quantize_and_render(events_with_theory, analysis_data)

    # 6. Generate MIDI bytes
    midi_bytes = b""
    tmp_xml_path = None
    tmp_midi_path = None

    try:
        # Save XML to temp file to parse back for MIDI generation
        # (music21 allows parsing string, but temp file is safer for some backends)
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False, mode='w', encoding='utf-8') as tmp_xml:
            tmp_xml.write(musicxml_str)
            tmp_xml_path = tmp_xml.name

        try:
            s = music21.converter.parse(tmp_xml_path)

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
                tmp_midi_path = tmp_midi.name

            s.write('midi', fp=tmp_midi_path)

            with open(tmp_midi_path, 'rb') as f:
                midi_bytes = f.read()

        except Exception as e:
            print(f"MIDI generation failed: {e}")
            midi_bytes = b""

    except Exception as e:
        print(f"Failed to generate MIDI wrapper: {e}")
    finally:
        if tmp_xml_path and os.path.exists(tmp_xml_path):
            os.remove(tmp_xml_path)
        if tmp_midi_path and os.path.exists(tmp_midi_path):
            os.remove(tmp_midi_path)

    result = TranscriptionResult(
        musicxml=musicxml_str,
        analysis_data=analysis_data,
        midi_bytes=midi_bytes
    )

    return result

# wrapper for legacy calls
def transcribe_audio(
    file_path: str,
    use_mock: bool = False,
    stereo_mode: bool = False,
) -> str:
    """
    Legacy entry point.
    """
    res = transcribe_audio_pipeline(file_path, use_mock=use_mock, stereo_mode=str(stereo_mode))
    return res.musicxml

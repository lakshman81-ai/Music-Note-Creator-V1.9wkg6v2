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
from .pipeline.models import AnalysisData, TranscriptionResult

def transcribe_audio_pipeline(
    audio_path: str,
    *,
    stereo_mode: Optional[str] = None, # ignored
    use_mock: bool = False, # ignored
    start_offset: Optional[float] = None, # ignored
    max_duration: Optional[float] = None, # ignored
    use_crepe: bool = False,
    **kwargs,
) -> Any: # Returns TranscriptionResult (which acts like a dict)
    """
    High-level API used by both FastAPI and the benchmark script.
    Orchestrates Stages A -> B -> C -> D.
    """
    if use_mock:
        # Legacy mock behavior for testing
        from .pipeline.models import AnalysisData, MetaData
        mock_xml_path = os.path.join(os.path.dirname(__file__), "mock_data", "happy_birthday.xml")
        if os.path.exists(mock_xml_path):
            with open(mock_xml_path, 'r', encoding='utf-8') as f:
                musicxml_str = f.read()
        else:
            musicxml_str = "<?xml ...>"

        return TranscriptionResult(
            musicxml=musicxml_str,
            analysis_data=AnalysisData(meta=MetaData(sample_rate=22050)),
            midi_bytes=b""
        )

    # 1. Stage A: Load and Preprocess
    # Robust loading and normalization
    y, sr, meta = load_and_preprocess(audio_path, target_sr=22050)

    # 2. Stage B: Extract Features (Segmentation)
    # Pitch tracking (pyin/crepe) and Hysteresis segmentation
    timeline, notes, chords = extract_features(y, sr, meta, use_crepe=use_crepe)

    # 3. Build AnalysisData
    analysis_data = AnalysisData(
        meta=meta,
        timeline=timeline,
        events=notes,
        chords=chords,
        notes=notes,
        pitch_tracker="crepe" if use_crepe else "pyin",
        n_frames=len(timeline),
        frame_hop_seconds=float(meta.hop_length) / float(meta.target_sr)
    )

    # Store pre-quantization notes
    # We need to deep copy if we want to preserve them exactly,
    # but NoteEvent is mutable. Stage D modifies them in place (adding beat info).
    # So let's copy them if we want to keep "raw" timing.
    # For now, just assigning. Stage D adds fields, doesn't change start_sec/end_sec usually,
    # but does sort them.
    from dataclasses import replace
    analysis_data.notes_before_quantization = [replace(n) for n in notes]

    # 4. Stage C: Apply Theory
    events_with_theory = apply_theory(notes, analysis_data)

    # 5. Stage D: Quantize and Render
    musicxml_str = quantize_and_render(events_with_theory, analysis_data)

    # 6. Generate MIDI bytes
    midi_bytes = b""
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

            os.remove(tmp_midi_path)

        except Exception as e:
            print(f"MIDI generation failed: {e}")
            midi_bytes = b""
        finally:
            if os.path.exists(tmp_xml_path):
                os.remove(tmp_xml_path)

    except Exception as e:
        print(f"Failed to generate MIDI wrapper: {e}")

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

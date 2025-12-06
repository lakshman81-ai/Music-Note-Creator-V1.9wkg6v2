from typing import Dict, Any, Optional
import io
import music21
import base64
import tempfile
import os

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
    """

    # 1. Load and Preprocess
    y, sr, meta = load_and_preprocess(audio_path, target_sr=22050)

    # 2. Extract Features
    timeline, notes, chords = extract_features(y, sr, meta, use_crepe=use_crepe)

    # 3. Build AnalysisData
    analysis_data = AnalysisData(
        meta=meta,
        timeline=timeline,
        events=notes,
        chords=chords,
        notes=notes # populate both for safety
    )

    # 4. Apply Theory
    events_with_theory = apply_theory(notes, analysis_data)

    # 5. Quantize and Render
    musicxml_str = quantize_and_render(events_with_theory, analysis_data)

    # 6. Generate MIDI bytes
    try:
        # music21.converter.parse might struggle with string directly if not handled perfectly.
        # Let's save to a temp file to be safe.
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False, mode='w', encoding='utf-8') as tmp_xml:
            tmp_xml.write(musicxml_str)
            tmp_xml_path = tmp_xml.name

        try:
            s = music21.converter.parse(tmp_xml_path)

            # Write MIDI to a temp file because music21.write('midi') expects a filename,
            # and writing to BytesIO is sometimes finicky with different versions or backends.
            # But let's try with fp=BytesIO first if we can fix the error.
            # The error was: "expected str, bytes or os.PathLike object, not BytesIO"
            # This suggests s.write('midi', fp=...) might not support fp argument in this version or calls something that doesn't.
            # Actually, `write` usually takes `fmt` and `fp`.
            # If `fp` is not a path string, it might fail depending on the backend.
            # Let's use a temp file for MIDI output.

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
                tmp_midi_path = tmp_midi.name

            # Closing the file so music21 can write to it

            s.write('midi', fp=tmp_midi_path)

            with open(tmp_midi_path, 'rb') as f:
                midi_bytes = f.read()

            os.remove(tmp_midi_path)

        finally:
            os.remove(tmp_xml_path)

    except Exception as e:
        print(f"Failed to generate MIDI: {e}")
        midi_bytes = b""

    result = TranscriptionResult(
        musicxml=musicxml_str,
        analysis_data=analysis_data,
        midi_bytes=midi_bytes
    )

    return result

# wrapper for legacy calls if any not covered
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

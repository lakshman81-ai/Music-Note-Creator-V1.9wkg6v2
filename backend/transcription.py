import os
import shutil
import tempfile
import asyncio
from typing import Optional

# Conditional imports for heavy libraries to allow verification in lightweight environments
try:
    import librosa
    import numpy as np
    from basic_pitch.inference import predict
    from music21 import stream, note, chord, meter, key, tempo, metadata, converter
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    print("Warning: ML dependencies not found. Only mock mode is available.")

def transcribe_audio(file_path: str, use_mock: bool = False) -> str:
    """
    Main entry point for the transcription pipeline.

    Args:
        file_path (str): Path to the input audio file (mp3/wav).
        use_mock (bool): If True, bypasses ML inference and returns mock MusicXML.

    Returns:
        str: The generated MusicXML content.
    """
    if use_mock or not HAS_ML_DEPS:
        print(f"Transcription: Using mock data (Reason: {'User Flag' if use_mock else 'Missing Deps'})")
        mock_path = os.path.join(os.path.dirname(__file__), "mock_data", "happy_birthday.xml")
        if not os.path.exists(mock_path):
            raise FileNotFoundError(f"Mock data not found at {mock_path}")
        with open(mock_path, "r", encoding="utf-8") as f:
            return f.read()

    # --- Real Pipeline ---
    try:
        # 1. Preprocessing (Source Separation / HPSS)
        # For V1, we use librosa HPSS to isolate harmonic content.
        # Ideally, we would use Spleeter or Demucs here if GPU is available.
        print(f"Loading audio: {file_path}")
        y, sr = librosa.load(file_path, sr=22050)
        y_harmonic, _ = librosa.effects.hpss(y)

        # Save temporary harmonic file for Basic Pitch
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            import soundfile as sf
            sf.write(tmp_wav.name, y_harmonic, sr)
            tmp_wav_path = tmp_wav.name

        # 2. Transcription (Basic Pitch)
        # predict returns: model_output, midi_data, note_events
        print("Running Basic Pitch inference...")
        _, midi_data, _ = predict(tmp_wav_path)

        # Cleanup temp file
        os.remove(tmp_wav_path)

        # 3. Post-processing & MusicXML Generation (Music21)
        # Basic Pitch returns a PrettyMIDI object (or similar structure depending on version)
        # We need to convert this to Music21 stream

        # Note: basic-pitch's `midi_data` is a pretty_midi object.
        # We can write it to a temp MIDI file and let music21 parse it,
        # which is often more robust than manual conversion.

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_mid:
            midi_data.write(tmp_mid.name)
            tmp_mid_path = tmp_mid.name

        print("Converting MIDI to MusicXML...")
        s = converter.parse(tmp_mid_path)

        # Apply quantization and cleanup
        s.quantize([4, 8, 16, 32], processOffsets=True, processDurations=True, inPlace=True)

        # Estimate Key
        k = s.analyze('key')
        s.insert(0, k)

        # Estimate Time Signature (if not already present from MIDI)
        # music21's midi parser usually guesses 4/4 if undefined.
        # We can run more advanced analysis here if needed.

        # Generate MusicXML string
        # GEX is music21's internal XML exporter
        from music21.musicxml import exporter
        sx = exporter.ScoreExporter(s)
        musicxml_str = sx.parse()

        # Cleanup
        os.remove(tmp_mid_path)

        return musicxml_str.decode('utf-8')

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

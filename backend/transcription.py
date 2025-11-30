import os
import shutil
import tempfile
import asyncio
from typing import Optional
import copy

# Conditional imports for heavy libraries to allow verification in lightweight environments
try:
    import librosa
    import numpy as np
    from basic_pitch.inference import predict
    from music21 import stream, note, chord, meter, key, tempo, metadata, converter, clef
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    print("Warning: ML dependencies not found. Only mock mode is available.")

def clean_and_format_score(midi_stream):
    """
    Transforms raw AI MIDI into 'Expected Output' style sheet music.
    """

    # 1. GHOST NOTE FILTERING (Must be done BEFORE Quantization)
    # Remove very short notes that are likely noise.
    notes_to_remove = []
    # Use recurse() or flatten() to get all notes.
    # Note: removing from a stream while iterating is unsafe, so we collect first.
    for n in midi_stream.recurse().notes:
        if n.duration.quarterLength < 0.1:
            notes_to_remove.append(n)

    for n in notes_to_remove:
        # We need to find the parent container to remove it safely
        try:
            # Try removing from activeSite first (immediate parent)
            if n.activeSite:
                n.activeSite.remove(n)
            else:
                midi_stream.remove(n, recurse=True)
        except Exception:
            pass

    # 2. QUANTIZATION
    # Snap to Quarter (1.0) and Eighth (0.5) notes.
    # We use [2] to enforce 8th note grid (which includes quarters).
    try:
        midi_stream.quantize([2], processOffsets=True, processDurations=True, inPlace=True)
    except Exception as e:
        print(f"Quantization warning: {e}")

    # 3. KEY CORRECTION (Force C Major / Strip Accidentals)
    # As per instructions: "Force 'accidental' notes to their nearest scale neighbor"
    # Simplification for V1: Force C Major (no accidentals).

    # Remove existing keys
    midi_stream.removeByClass('Key')
    midi_stream.removeByClass('KeySignature')

    # Insert C Major
    c_major = key.Key('C')
    midi_stream.insert(0, c_major)

    # Force all notes to be natural (strip accidentals)
    for n in midi_stream.recurse().notes:
        if isinstance(n, note.Note):
            if n.pitch.accidental is not None:
                n.pitch.accidental = None
        elif isinstance(n, chord.Chord):
            for p in n.pitches:
                p.accidental = None

    # 4. GRAND STAFF SEPARATION
    # Create the container score
    score = stream.Score()

    # Create Right Hand (Treble)
    right_hand = stream.Part()
    right_hand.id = 'RightHand'
    right_hand.partName = 'Treble'
    right_hand.append(clef.TrebleClef())

    # Create Left Hand (Bass)
    left_hand = stream.Part()
    left_hand.id = 'LeftHand'
    left_hand.partName = 'Bass'
    left_hand.append(clef.BassClef())

    # Iterate through notes and assign based on pitch (Split point: Middle C / C4 / MIDI 60)
    # We iterate over the flat stream to preserve absolute timing.
    source_notes = []
    for el in midi_stream.flatten().notes:
        source_notes.append(el)

    for n in source_notes:
        # Determine split
        is_treble = True
        if isinstance(n, note.Note):
            if n.pitch.midi < 60:
                is_treble = False
        elif isinstance(n, chord.Chord):
            # Check average pitch
            avg_midi = sum(p.midi for p in n.pitches) / len(n.pitches)
            if avg_midi < 60:
                is_treble = False

        # Insert into appropriate part
        # When moving from one stream to another, we need the absolute offset.
        offset = n.getOffsetInHierarchy(midi_stream)

        # Create a deep copy to insert into the new Score
        new_n = copy.deepcopy(n)

        if is_treble:
            right_hand.insert(offset, new_n)
        else:
            left_hand.insert(offset, new_n)

    # Add parts to score
    score.insert(0, right_hand)
    score.insert(0, left_hand)

    # 5. FINAL LAYOUT
    score.makeNotation(inPlace=True)

    return score

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

        # --- NEW CLEANING PIPELINE ---
        print("Running Cleaning Pipeline...")
        cleaned_score = clean_and_format_score(s)

        # Generate MusicXML string
        # GEX is music21's internal XML exporter
        from music21.musicxml import exporter
        sx = exporter.ScoreExporter(cleaned_score)
        musicxml_str = sx.parse()

        # Cleanup
        os.remove(tmp_mid_path)

        return musicxml_str.decode('utf-8')

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

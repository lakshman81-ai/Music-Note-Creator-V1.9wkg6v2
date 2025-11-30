from typing import List, Dict
import math
from backend.pipeline.models import NoteEvent, ChordEvent, AnalysisData, MetaData, VexflowLayout
from music21 import stream, note, chord, meter, key, layout, dynamics, clef, tempo, duration, metadata
from music21.musicxml import m21ToXml

def quantize_and_render(events: List[NoteEvent], analysis_data: AnalysisData) -> str:
    """
    Stage D: Rational Quantization & Notation

    1. Multi-Grid Quantization (Straight vs Swing/Triplet)
    2. Tuplet Detection
    3. Beaming Rules
    4. Audio Sync Mapping
    5. Generate MusicXML
    """

    # 0. Setup Music21 Score
    s = stream.Score()

    # Setup Metadata
    s.insert(0, metadata.Metadata())
    s.metadata.title = "Transcribed Score"

    # Setup Parts (Treble and Bass)
    p_treble = stream.Part()
    p_treble.id = 'P1'
    p_treble.partName = 'Treble'
    p_treble.insert(0, clef.TrebleClef())

    p_bass = stream.Part()
    p_bass.id = 'P2'
    p_bass.partName = 'Bass'
    p_bass.insert(0, clef.BassClef())

    # Key Signature
    k = key.Key(analysis_data.meta.detected_key)
    p_treble.insert(0, k)
    p_bass.insert(0, k)

    # Time Signature (Default 4/4)
    ts = meter.TimeSignature('4/4')
    p_treble.insert(0, ts)
    p_bass.insert(0, ts)

    # BPM (Estimate or Default)
    bpm = 120 # Placeholder, ideally detected
    mm = tempo.MetronomeMark(number=bpm)
    p_treble.insert(0, mm)
    p_bass.insert(0, mm)

    # D1. Multi-Grid Quantization
    # We need to assign start_beat and duration_beat to each event.
    # Convert sec to beat: beat = sec * (BPM / 60)

    # Simple quantization logic
    def quantize_beat(b, grid_choices=[0.25, 1/3]):
        # Find nearest grid point
        # grid_choices: 0.25 (16th), 1/3 (triplet 8th)
        best_diff = float('inf')
        best_q = b

        for grid in grid_choices:
            # Snap b to nearest multiple of grid
            rounded = round(b / grid) * grid
            diff = abs(b - rounded)
            if diff < best_diff:
                best_diff = diff
                best_q = rounded
        return best_q

    # "Select grid with lowest quantization error."
    # We should analyze the whole sequence to decide if it's swing (triplet) or straight.
    # Calculate error for straight grid (0.25) vs triplet grid (1/3)
    error_straight = 0
    error_swing = 0

    for e in events:
        raw_beat = e.start_sec * (bpm / 60)

        # Straight
        q_str = round(raw_beat / 0.25) * 0.25
        error_straight += abs(raw_beat - q_str)

        # Swing/Triplet (using 1/3 for 8th triplets)
        # Note: Swing is often approximated as triplets.
        q_swi = round(raw_beat / (1/3)) * (1/3)
        error_swing += abs(raw_beat - q_swi)

    use_triplets = error_swing < error_straight * 0.8 # Bias towards straight unless swing is clear

    grid_res = 1/3 if use_triplets else 0.25

    for e in events:
        # Calculate beats
        raw_start_beat = e.start_sec * (bpm / 60)
        raw_duration_beat = (e.end_sec - e.start_sec) * (bpm / 60)

        # Snap
        e.start_beat = round(raw_start_beat / grid_res) * grid_res
        e.duration_beat = max(grid_res, round(raw_duration_beat / grid_res) * grid_res)

        # Create Music21 Note
        m21_note = note.Note(e.midi_note)
        m21_note.quarterLength = e.duration_beat

        # Dynamics
        if e.dynamic:
            m21_note.dynamics = dynamics.Dynamic(e.dynamic)

        # Grace Notes
        if e.type == "grace":
            m21_note.duration.quarterLength = 0.0
            m21_note.getGrace(inPlace=True)

        # Insert into correct part (Voice)
        # We used Voice 1 (Treble) and Voice 2 (Bass) in Stage B
        if e.voice == 1:
            p_treble.insert(e.start_beat, m21_note)
        else:
            p_bass.insert(e.start_beat, m21_note)

    # D3. Beaming Rules & D2. Tuplets
    # music21 handles beaming automatically with makeNotation()
    # It also handles tuplet detection if quarterLengths are rational (like 1/3)

    p_treble.makeNotation(inPlace=True)
    p_bass.makeNotation(inPlace=True)

    s.insert(0, p_treble)
    s.insert(0, p_bass)

    # D4. Audio Sync Mapping
    # Update events with final timings from the score?
    # Or just rely on our quantization.
    # The requirement says: "Every note must store start_sec, end_sec, start_beat, duration_beat"
    # We have already populated NoteEvent with these.

    # Generate XML
    gex = m21ToXml.GeneralObjectExporter(s)
    xml_str = gex.parse().decode('utf-8')

    # Populate Vexflow Layout (Logical)
    # Extract measure info
    measures = []
    # Iterate measures in treble part
    for m in p_treble.getElementsByClass('Measure'):
        measures.append({
            "number": m.number,
            "width": 0, # Placeholder
            # Add more info if needed
        })
    analysis_data.vexflow_layout = VexflowLayout(measures=measures)

    return xml_str

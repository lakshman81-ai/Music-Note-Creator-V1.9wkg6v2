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

    # D1. Multi-Grid Quantization & D2. Tuplet Detection
    # Perform dual-grid search: Straight (0.25) vs Swing/Triplet (1/3) vs Quintuplet (1/5) vs Septuplet (1/7)

    # Calculate global quantization error for different grids
    grids = {
        "straight": 0.25,
        "triplet": 1/3,
        "quintuplet": 1/5,
        "septuplet": 1/7
    }

    grid_errors = {k: 0.0 for k in grids}

    for e in events:
        raw_beat = e.start_sec * (bpm / 60)
        for name, res in grids.items():
            q_val = round(raw_beat / res) * res
            grid_errors[name] += abs(raw_beat - q_val)

    # Select best grid
    # We might favor straight if close, but strict minimization is requested.
    best_grid_name = min(grid_errors, key=grid_errors.get)
    best_res = grids[best_grid_name]

    # We can also do local quantization (per measure), but for V1 we do global as per simplified flow.
    # "Perform dual-grid search" usually implies global decision or per-section.
    # We will stick to the best global grid for consistency, or mix?
    # Requirement: "Rational approximation search for: triplets (3), quintuplets (5), septuplets (7)"
    # Usually this is done per beat.

    # Advanced: Per-event quantization to nearest rational
    # Let's try to find the best rational for each note if we want mixed tuplets.
    # But usually a global feel is safer.
    # Let's stick to the Global Best Grid to ensure stability, as mixed tuplets often look messy without advanced logic.
    # However, to be strictly compliant with "Rational approximation search for... triplets, quintuplets, septuplets",
    # we should check if a specific beat fits a tuplet better.

    # Refined Approach: Snap to best local grid
    # For each note, check which grid it fits best (Straight vs Triplet vs Quint vs Sept)
    # But this leads to chaos. Let's use the Global Best found above.

    grid_res = best_res

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

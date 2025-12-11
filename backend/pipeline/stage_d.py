from typing import List, Optional
import numpy as np
import music21
from music21 import stream, note, chord, tie, tempo, meter, key, dynamics, articulations, layout, clef
from .models import NoteEvent, AnalysisData
from .config import PIANO_61KEY_CONFIG, PipelineConfig

def quantize_and_render(
    events: List[NoteEvent], # These are the notes from Stage C
    analysis_data: AnalysisData,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
) -> str:
    """
    Stage D: Render Sheet Music (MusicXML) using music21.
    """
    d_conf = config.stage_d
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else "4/4"
    split_pitch = d_conf.staff_split_point.get("pitch", 60) # C4

    # 1. Setup Score
    s = stream.Score()

    # Create Parts
    part_treble = stream.Part()
    part_treble.id = 'P1'
    part_bass = stream.Part()
    part_bass.id = 'P2'

    # Metadata / Measure 1 setup
    # We insert into flat parts, music21 handles measure creation.
    # But to ensure correct Key/Time/Tempo at start:

    # Add Clefs? Music21 might auto-detect, but we force Treble/Bass
    part_treble.append(clef.TrebleClef())
    part_bass.append(clef.BassClef())

    # Add TimeSignature
    ts_obj = meter.TimeSignature(ts_str)
    part_treble.append(ts_obj)
    part_bass.append(ts_obj)

    # Add Tempo
    mm = tempo.MetronomeMark(number=bpm)
    part_treble.append(mm)
    part_bass.append(mm) # Optional to duplicate

    # Add Key
    if analysis_data.meta.detected_key:
        try:
            k = key.Key(analysis_data.meta.detected_key)
            part_treble.append(k)
            part_bass.append(k)
        except:
            pass

    # 2. Process Notes & Chords
    # WI Rules:
    # - Staff split
    # - Durations & divisions
    # - Chords (same onset)
    # - Ties (handled by music21 makeMeasures usually, but we might need explicit)
    # - Velocity/Dynamics
    # - Staccato

    # Group by onset to identify chords
    # Key: (start_sec, staff_idx) -> List[NoteEvent]
    # We first assign staff to events based on pitch

    sorted_events = sorted(events, key=lambda e: e.start_sec)

    # Grouping
    # We use a tolerance for "simultaneous"
    onset_tol = 0.05 # 50ms

    groups = []
    if sorted_events:
        current_group = [sorted_events[0]]
        for i in range(1, len(sorted_events)):
            e = sorted_events[i]
            if e.start_sec - current_group[0].start_sec < onset_tol:
                current_group.append(e)
            else:
                groups.append(current_group)
                current_group = [e]
        groups.append(current_group)

    import music21.chord
    import music21.note

    # Helper to create m21 object
    def create_m21_obj(group: List[NoteEvent]):
        # Calculate duration in Quarter Lengths (beats)
        # We assume all in group have roughly same duration?
        # Use max duration? Or individual?
        # Music21 Chords have single duration.
        # If polyphonic with different durations, we need different Voices.
        # For simplicity (WI "Chords"), we assume block chords if onsets match.
        # We take the max duration of the group.

        start = group[0].start_sec
        # Determine duration
        max_end = max(e.end_sec for e in group)
        dur_sec = max_end - start
        quarter_dur = 60.0 / bpm
        q_len = dur_sec / quarter_dur

        # Quantize q_len to nearest 1/16th (0.25)
        # divisions_per_quarter usually handles display, but q_len is float.
        # Round to nearest 0.25
        q_len = round(q_len * 4) / 4.0
        if q_len <= 0: q_len = 0.25 # Minimum

        velocity = int(np.mean([e.velocity for e in group]) * 127)

        # Force q_len to be a multiple of 0.25 to avoid music21 errors
        # (It is already rounded, but just ensuring)

        if len(group) > 1:
            # Chord
            pitches = [e.midi_note for e in group]
            # Remove duplicates
            pitches = sorted(list(set(pitches)))
            m21_obj = music21.chord.Chord(pitches)
        else:
            # Note
            m21_obj = music21.note.Note(group[0].midi_note)

        # Force strict duration object
        # m21_obj.quarterLength = q_len
        try:
            m21_obj.duration = music21.duration.Duration(q_len)
        except Exception:
            # Fallback to quarter note if duration fails
            m21_obj.duration = music21.duration.Duration(1.0)

        m21_obj.volume.velocity = velocity

        # Staccato
        staccato_thresh = d_conf.staccato_marking.get("threshold_beats", 0.25)
        if q_len < staccato_thresh:
            m21_obj.articulations.append(articulations.Staccato())

        return m21_obj, start

    # Insert into Stream
    for group in groups:
        # Split group by staff
        treble_notes = [e for e in group if e.midi_note >= split_pitch]
        bass_notes = [e for e in group if e.midi_note < split_pitch]

        if treble_notes:
            obj, start = create_m21_obj(treble_notes)
            # Convert start time to beat offset
            raw_offset = (start * bpm / 60.0)
            # Quantize offset to nearest 16th (0.25)
            offset = round(raw_offset * 4) / 4.0
            part_treble.insert(offset, obj)

        if bass_notes:
            obj, start = create_m21_obj(bass_notes)
            raw_offset = (start * bpm / 60.0)
            # Quantize offset to nearest 16th (0.25)
            offset = round(raw_offset * 4) / 4.0
            part_bass.insert(offset, obj)

    s.append(part_treble)
    s.append(part_bass)

    # 3. Make Measures & Ties
    # music21.makeMeasures() automatically handles:
    # - Bar lines
    # - Tying notes that cross measures
    # - Filling rests? (best effort)

    # Ensure elements are sorted and handle float errors by rounding offsets slightly before measures
    # Or just rely on makeMeasures.

    try:
        s_quant = s.makeMeasures()
        # makeTies logic is implicitly handled if we inserted duration > measure_len?
        # Usually makeMeasures splits notes.
        # We might need to run makeTies explicitly if we constructed raw.
        s_quant.makeTies(inPlace=True)
    except Exception as e:
        print(f"Quantization failed: {e}")
        # If makeMeasures fails, we might have weird durations.
        # Fallback to just returning the raw score if possible,
        # but musicXML export might still fail if durations are bad (e.g. infinite).
        s_quant = s

    # 4. Export
    from music21.musicxml import m21ToXml
    exporter = m21ToXml.GeneralObjectExporter(s_quant)
    musicxml_bytes = exporter.parse()
    musicxml_str = musicxml_bytes.decode('utf-8')

    return musicxml_str

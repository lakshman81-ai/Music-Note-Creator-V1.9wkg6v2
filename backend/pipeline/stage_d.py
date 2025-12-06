from typing import List
from .models import NoteEvent, AnalysisData
from music21 import stream, note, tempo, meter, metadata, key

def quantize_and_render(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> str:
    """
    Stage D: Quantize NoteEvents into beats/measures and render to MusicXML string.
    """
    # 1. Get Tempo and Time Signature
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else "4/4"

    # 2. Quantize
    # "Use a simple 1/16-note grid"
    quarter_sec = 60.0 / bpm
    sixteenth_sec = quarter_sec / 4.0
    beats_per_measure = 4 # Assuming 4/4 for now as per requirements logic

    # Simple quantization logic
    for e in events:
        # Quantize start and end
        # Snap to nearest sixteenth
        start_grid = round(e.start_sec / sixteenth_sec)
        end_grid = round(e.end_sec / sixteenth_sec)

        if end_grid <= start_grid:
            end_grid = start_grid + 1 # Ensure at least 1/16th duration

        duration_grid = end_grid - start_grid

        # Convert grid units back to beats (1 grid unit = 0.25 beats)
        start_beat_global = start_grid * 0.25
        duration_beats = duration_grid * 0.25

        e.duration_beats = duration_beats

        # Calculate measure and beat within measure
        # 4/4 time -> 4 beats per measure
        e.measure = int(start_beat_global // beats_per_measure) + 1 # Measures are 1-indexed usually
        e.beat = (start_beat_global % beats_per_measure) + 1.0 # Beats are 1-indexed

    # 3. Build music21 Score
    s = stream.Score()
    s.metadata = metadata.Metadata()
    s.metadata.title = "Analyzed Audio"

    p = stream.Part()

    # Add TimeSignature
    m1 = stream.Measure()
    m1.number = 1
    ts = meter.TimeSignature(ts_str)
    m1.timeSignature = ts

    # Add MetronomeMark
    mm = tempo.MetronomeMark(number=bpm)
    m1.append(mm)

    # Add KeySignature if available
    # analysis_data.meta.detected_key is "C" default.
    # music21 KeySignature takes sharps count or key name.
    # music21.key.Key('C') works.
    if analysis_data.meta.detected_key:
        try:
            k = key.Key(analysis_data.meta.detected_key)
            m1.append(k)
        except Exception:
            pass # Fallback to no key signature (C major / A minor implied)

    # We need to distribute notes into measures.
    # Music21 can handle a flat stream and `makeMeasures()`, or we can manually place them.
    # Since we calculated measure numbers, we could try to place them manually,
    # but `makeMeasures` is more robust for handling tied notes across bar lines.
    # Let's create a flat stream of notes with correct offsets and let music21 handle the layout.

    # Create notes
    # We will insert them into the Part at the correct offsets (in beats).
    # Since we calculated `start_beat_global` (implicitly via `measure` and `beat`),
    # we can reconstruct the offset.
    # offset = (measure - 1) * 4 + (beat - 1)

    for e in events:
        if e.midi_note is None:
            continue

        n = note.Note(e.midi_note)
        n.quarterLength = e.duration_beats

        # Calculate offset
        offset = (e.measure - 1) * beats_per_measure + (e.beat - 1)

        # Insert into part (flat)
        p.insert(offset, n)

    # Make measures
    # This automatically splits notes across bar lines if needed and creates measures.
    try:
        # Ensure the part has the time signature at the beginning for makeMeasures to work correctly
        p.insert(0, ts)
        p_measures = p.makeMeasures()
        # Transfer metadata and other elements if needed, but makeMeasures returns a stream with measures.
        # We need to put this back into the score.
        s.append(p_measures)
    except Exception as e:
        print(f"music21 makeMeasures failed: {e}. Returning flat part.")
        s.append(p)

    # 4. Export to MusicXML
    # music21.musicxml.m21ToXml.GeneralObjectExporter
    from music21.musicxml import m21ToXml
    exporter = m21ToXml.GeneralObjectExporter(s)
    musicxml_bytes = exporter.parse() # This returns bytes usually
    musicxml_str = musicxml_bytes.decode('utf-8')

    # 5. Populate vexflow_layout (simple list of measure numbers)
    # Just a list of indices or similar as requested "do not make it complex"
    if analysis_data.vexflow_layout:
        # Just count measures
        try:
            num_measures = len(p_measures.getElementsByClass(stream.Measure))
            analysis_data.vexflow_layout.measures = [{"index": i} for i in range(num_measures)]
        except:
             analysis_data.vexflow_layout.measures = []

    return musicxml_str

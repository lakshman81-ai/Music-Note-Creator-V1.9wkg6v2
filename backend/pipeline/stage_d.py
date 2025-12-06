from typing import List
from .models import NoteEvent, AnalysisData
from music21 import stream, note, tempo, meter, key, layout

def quantize_and_render(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> str:
    """
    Stage D: Quantize NoteEvents into strict 1/16th grid and render to MusicXML string.
    """
    # 1. Setup global context
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else "4/4"

    quarter_sec = 60.0 / bpm
    sixteenth_sec = quarter_sec / 4.0
    beats_per_measure = 4 # assume 4/4 for calculation logic

    # 2. Quantize
    # Strict 1/16th grid.

    # Sort events by time just in case
    events.sort(key=lambda x: x.start_sec)

    quantized_notes = []

    for e in events:
        # Snap start and end to nearest 16th grid index
        start_idx = round(e.start_sec / sixteenth_sec)
        end_idx = round(e.end_sec / sixteenth_sec)

        # Ensure minimum duration of 1/16th
        if end_idx <= start_idx:
            end_idx = start_idx + 1

        duration_idx = end_idx - start_idx

        # Convert back to beats (1 unit = 0.25 beats)
        start_beat_global = start_idx * 0.25
        duration_beats = duration_idx * 0.25

        e.duration_beats = duration_beats

        # Calculate measure and beat (1-based)
        e.measure = int(start_beat_global // beats_per_measure) + 1
        e.beat = (start_beat_global % beats_per_measure) + 1.0

        quantized_notes.append((start_beat_global, e))

    # 3. Build Music21 Score
    s = stream.Score()
    p = stream.Part()

    # Setup Measure 1
    m1 = stream.Measure()
    m1.number = 1
    m1.timeSignature = meter.TimeSignature(ts_str)
    m1.append(tempo.MetronomeMark(number=bpm))

    if analysis_data.meta.detected_key:
        try:
            m1.append(key.Key(analysis_data.meta.detected_key))
        except:
            pass

    # Instead of inserting into m1, we insert into the Part at specific offsets
    # and let makeMeasures handle the rest.
    # Note: Inserting m1 into p first might define the context.

    p.insert(0, m1.timeSignature)
    p.insert(0, m1.getElementsByClass(tempo.MetronomeMark).first())
    if m1.keySignature:
        p.insert(0, m1.keySignature)

    for start_beat, e in quantized_notes:
        if e.midi_note is None:
            continue

        n = note.Note(e.midi_note)
        n.quarterLength = e.duration_beats
        n.volume.velocity = int(e.velocity * 127) if e.velocity else 80

        # Insert into flat part
        p.insert(start_beat, n)

    # 4. Make Measures
    # makeMeasures() splits notes across bar lines and handles rests (if we used makeRests=True, but
    # inserting into flat stream usually leaves gaps as implicit rests or specific behavior).
    # music21's makeMeasures is best effort.

    try:
        s_measures = p.makeMeasures()
        s.append(s_measures)
    except Exception as e:
        print(f"makeMeasures failed: {e}. Returning flat.")
        s.append(p)

    # 5. Export
    from music21.musicxml import m21ToXml
    exporter = m21ToXml.GeneralObjectExporter(s)
    musicxml_bytes = exporter.parse()
    musicxml_str = musicxml_bytes.decode('utf-8')

    # 6. Update Layout info
    if analysis_data.vexflow_layout:
        try:
            # Count measures
            if s.hasMeasures():
                # Get the part
                part = s.parts[0]
                measures = part.getElementsByClass(stream.Measure)
                analysis_data.vexflow_layout.measures = [{"index": m.number} for m in measures]
        except:
            analysis_data.vexflow_layout.measures = []

    return musicxml_str

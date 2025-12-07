from typing import List, Optional
import numpy as np
from music21 import stream, note, tempo, meter, key, dynamics, articulations
from .models import NoteEvent, AnalysisData

def quantize_and_render(
    events: List[NoteEvent], # These are already quantized in Stage C, but we do final formatting here
    analysis_data: AnalysisData,
) -> str:
    """
    Stage D: Render Sheet Music (MusicXML/MIDI)
    """
    # 1. Setup global context
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else "4/4"

    # Gap Merging
    # "Gap merging for notes separated by < 1/32 note"
    # 1/32 note = 1/8 beat
    gap_threshold_beats = 0.125

    merged_events = []
    if events:
        events.sort(key=lambda x: x.start_sec)
        current_note = events[0]

        for i in range(1, len(events)):
            next_note = events[i]

            # Calculate gap in beats
            # We need to rely on the quantized start/end from Stage C
            # but Stage C updated .start_sec and .end_sec to match grid.
            # So we can compare seconds or beats.

            gap_sec = next_note.start_sec - current_note.end_sec
            quarter_sec = 60.0 / bpm
            gap_beats = gap_sec / quarter_sec

            # Check overlap? "No overlapping same-pitch notes"
            if next_note.midi_note == current_note.midi_note:
                if gap_beats < gap_threshold_beats:
                    # Merge
                    current_note.end_sec = next_note.end_sec
                    # Update duration beats
                    if current_note.duration_beats is None:
                         current_note.duration_beats = 0.0
                    if next_note.duration_beats is None:
                         next_note.duration_beats = 0.0

                    current_note.duration_beats += next_note.duration_beats + gap_beats
                    # Recalculate duration beats from seconds to be safe?
                    # duration_beats = (end - start) / quarter_sec
                    current_note.duration_beats = (current_note.end_sec - current_note.start_sec) / quarter_sec
                else:
                    merged_events.append(current_note)
                    current_note = next_note
            else:
                merged_events.append(current_note)
                current_note = next_note
        merged_events.append(current_note)

    # 2. Velocity Mapping
    # v = 20 + 85 * clamp((rms_dB – rms_min)/(rms_max – rms_min), 0, 1)
    # rms_min = -40 dB, rms_max = -4 dB

    rms_min = -40.0
    rms_max = -4.0

    final_notes_for_xml = []

    for e in merged_events:
        # Calculate dB
        if e.rms_value <= 0:
            val_db = -80.0
        else:
            val_db = 20 * np.log10(e.rms_value)

        # Clamp
        clamped = max(rms_min, min(rms_max, val_db))
        ratio = (clamped - rms_min) / (rms_max - rms_min)

        v = 20 + 85 * ratio
        midi_vel = int(round(v))
        e.velocity = midi_vel / 127.0

        final_notes_for_xml.append((e, midi_vel))

    # 3. Render MusicXML
    s = stream.Score()
    p = stream.Part()

    # Metadata
    m1 = stream.Measure()
    m1.number = 1

    try:
        ts_obj = meter.TimeSignature(ts_str)
    except:
        ts_obj = meter.TimeSignature("4/4")

    m1.timeSignature = ts_obj
    m1.append(tempo.MetronomeMark(number=bpm))

    if analysis_data.meta.detected_key:
        try:
            m1.append(key.Key(analysis_data.meta.detected_key))
        except:
            pass

    p.append(m1)

    # We need to insert notes at specific offsets.
    # Music21 handles measures if we use .makeMeasures() or we insert into a flat stream and then make measures.
    # Inserting into flat stream is easier.

    for e, vel in final_notes_for_xml:
        if e.midi_note is None: continue

        n = note.Note(e.midi_note)
        if e.duration_beats and e.duration_beats > 0:
            n.quarterLength = e.duration_beats
        else:
            n.quarterLength = 0.25 # Default fallback

        n.volume.velocity = vel

        # Staccato check
        # "Durations < 30 ticks -> staccato notation"
        # 30 ticks = 1/16th note = 0.25 beats
        if n.quarterLength < 0.25:
             n.articulations.append(articulations.Staccato())

        # Insert at absolute beat position
        # Stage C calculated `measure` and `beat`.
        # Absolute beat = (measure - 1) * 4 + (beat - 1) (assuming 4/4)
        # Better to rely on `start_sec` converted to beats

        abs_beat = (e.start_sec / (60.0/bpm))
        p.insert(abs_beat, n)

    # Make measures
    try:
        s.append(p)
        s = s.makeMeasures()
    except Exception as e:
        print(f"makeMeasures failed: {e}")

    # Export
    from music21.musicxml import m21ToXml
    exporter = m21ToXml.GeneralObjectExporter(s)
    musicxml_bytes = exporter.parse()
    musicxml_str = musicxml_bytes.decode('utf-8')

    # Update Layout
    if analysis_data.vexflow_layout:
         try:
             # Just count measures for basic layout
             measures = s.parts[0].getElementsByClass(stream.Measure)
             analysis_data.vexflow_layout.measures = [{"index": m.number} for m in measures]
         except:
             analysis_data.vexflow_layout.measures = []

    return musicxml_str

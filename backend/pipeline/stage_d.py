from typing import List, Optional
import numpy as np
from music21 import stream, note, tempo, meter, key
from .models import NoteEvent, AnalysisData

def quantize_and_render(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> str:
    """
    Stage D: Quantize NoteEvents into strict 1/16th grid, merge gaps, map velocity, and render to MusicXML.
    """
    # 1. Setup global context
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else "4/4"

    # Parse Time Signature for dynamic beats per measure
    try:
        ts_obj = meter.TimeSignature(ts_str)
        beats_per_measure = ts_obj.barDuration.quarterLength
    except:
        ts_obj = meter.TimeSignature("4/4")
        beats_per_measure = 4.0

    quarter_sec = 60.0 / bpm
    sixteenth_sec = quarter_sec / 4.0

    # 2. Sort events
    events.sort(key=lambda x: x.start_sec)

    # 3. Gap Merging
    # "Gap merging for notes separated by < 1/32 note"
    # 1/32 note duration in seconds = sixteenth_sec / 2
    gap_threshold = sixteenth_sec / 2.0

    merged_events = []
    if events:
        current_note = events[0]
        for i in range(1, len(events)):
            next_note = events[i]

            gap = next_note.start_sec - current_note.end_sec

            # Check if gap is small AND (optional but logical) it's the same pitch?
            # Requirement says "Gap merging for notes separated by < 1/32 note".
            # Usually this implies legato handling or fixing segmentation jitter.
            # If pitches are different, we shouldn't merge them into one note unless we are doing monophonic smoothing?
            # But NoteEvents are distinct.
            # If the requirement means "Closing gaps between notes", it implies extending the previous note's end time.
            # It does NOT say "merge two notes into one". It says "Gap merging".
            # I will assume: Extend end of previous note if gap is small.

            if 0 < gap < gap_threshold:
                # Extend current note
                current_note.end_sec = next_note.start_sec

            merged_events.append(current_note)
            current_note = next_note
        merged_events.append(current_note)
    else:
        merged_events = []

    # 4. Velocity Mapping
    # RMS (dB) -> MIDI [20, 105]
    # Input RMS is in linear scale in NoteEvent.rms_value (from Stage B update)
    # We need to convert linear to dB first.

    min_vel = 20
    max_vel = 105
    min_db = -60.0
    max_db = 0.0 # RMS of full scale sine is -3dB, but let's assume 0 is peak

    quantized_notes = []

    for e in merged_events:
        # Calculate Velocity
        if e.rms_value <= 0:
            db = -80.0
        else:
            db = 20 * np.log10(e.rms_value)

        # Clamp dB
        if db < min_db:
            midi_vel = min_vel
        elif db > max_db:
            midi_vel = max_vel
        else:
            # Linear map
            ratio = (db - min_db) / (max_db - min_db)
            midi_vel = min_vel + ratio * (max_vel - min_vel)

        e.velocity = midi_vel / 127.0 # Store as 0-1 float for internal consistency if needed
        final_midi_vel = int(round(midi_vel))

        # Quantize to 1/16th grid
        start_idx = round(e.start_sec / sixteenth_sec)
        end_idx = round(e.end_sec / sixteenth_sec)

        if end_idx <= start_idx:
            end_idx = start_idx + 1

        duration_idx = end_idx - start_idx

        start_beat_global = start_idx * 0.25
        duration_beats = duration_idx * 0.25

        e.duration_beats = duration_beats
        e.measure = int(start_beat_global // beats_per_measure) + 1
        e.beat = (start_beat_global % beats_per_measure) + 1.0

        quantized_notes.append((start_beat_global, e, final_midi_vel))

    # 5. Build Music21 Score
    s = stream.Score()
    p = stream.Part()

    # Use makeMeasures approach, but first populate a flat stream
    # Insert Metadata
    m1 = stream.Measure() # Dummy for context
    m1.timeSignature = ts_obj
    m1.append(tempo.MetronomeMark(number=bpm))

    if analysis_data.meta.detected_key:
        try:
            m1.append(key.Key(analysis_data.meta.detected_key))
        except:
            pass

    p.insert(0, m1.timeSignature)
    p.insert(0, m1.getElementsByClass(tempo.MetronomeMark).first())
    if m1.keySignature:
        p.insert(0, m1.keySignature)

    for start_beat, e, vel in quantized_notes:
        if e.midi_note is None:
            continue

        n = note.Note(e.midi_note)
        n.quarterLength = e.duration_beats
        n.volume.velocity = vel

        p.insert(start_beat, n)

    try:
        s.append(p.makeMeasures())
    except Exception as e:
        print(f"makeMeasures failed: {e}. Returning flat.")
        s.append(p)

    # 6. Export
    from music21.musicxml import m21ToXml
    exporter = m21ToXml.GeneralObjectExporter(s)
    musicxml_bytes = exporter.parse()
    musicxml_str = musicxml_bytes.decode('utf-8')

    # 7. Update Layout info
    if analysis_data.vexflow_layout:
        try:
             # Just count measures for basic layout
             measures = s.parts[0].getElementsByClass(stream.Measure)
             analysis_data.vexflow_layout.measures = [{"index": m.number} for m in measures]
        except:
             analysis_data.vexflow_layout.measures = []

    return musicxml_str

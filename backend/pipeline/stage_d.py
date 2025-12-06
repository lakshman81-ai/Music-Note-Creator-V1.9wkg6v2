from typing import List
from .models import NoteEvent, AnalysisData
from music21 import stream, note, tempo, meter, key, layout

def quantize_and_render(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> str:
    """
    Stage D: Quantize NoteEvents into strict 1/16th grid and render to MusicXML string.
    WI Compliant.
    """
    # 1. Setup global context
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else "4/4"

    quarter_sec = 60.0 / bpm
    sixteenth_sec = quarter_sec / 4.0

    try:
        # Parse time signature string like "4/4", "3/4", "6/8"
        num_str, den_str = ts_str.split('/')
        numerator = int(num_str)
        denominator = int(den_str)
        # Beats per measure usually equals numerator for simple time (x/4),
        # but for compound (6/8), it's numerator/3 * (3/2)?
        # For this quantization logic (1/16th grid), we treat everything as
        # quarter note beats (x/4) or handle denominator scaling.
        # music21's TimeSignature object handles this well, but we need a simple float here.
        # Let's assume the beat is a quarter note for "beat" calculation if denominator is 4.
        # If denominator is 8, beat is eighth?
        # Standard MIDI/DAW logic often keeps "beat" = quarter note.
        # Let's stick to simple parsing: numerator * (4 / denominator) beats per measure?
        # e.g. 4/4 -> 4 beats. 3/4 -> 3 beats. 6/8 -> 6 eighths -> 3 quarters?
        # For simplicity and robustness given we output 4/4 mostly:
        beats_per_measure = float(numerator) * (4.0 / float(denominator))
    except Exception:
        beats_per_measure = 4.0

    # 2. Quantize (WI: "onset/offset = nearest 1/16th")
    events.sort(key=lambda x: x.start_sec)

    quantized_notes_temp = []

    # WI: Map RMS velocity to MIDI range [20, 105]
    # We assume e.velocity holds RMS (0 to ~1.0, maybe less).
    # We need to find global max RMS to normalize? Or just assume standard range?
    # RMS of normalized audio (-23 LUFS) is around -23dBFS ~ 0.07.
    # A loud section might be -10dBFS ~ 0.3.
    # Let's do dynamic scaling or fixed scaling.
    # WI says "Velocity proportional to RMS energy".
    # Let's map [0, 0.3] -> [20, 105]?
    # Or normalize by the max RMS found in the events.

    max_rms = 0.001
    for e in events:
        if e.velocity > max_rms:
            max_rms = e.velocity

    for e in events:
        # Snap
        start_idx = round(e.start_sec / sixteenth_sec)
        end_idx = round(e.end_sec / sixteenth_sec)

        # WI: Enforce at least 1 grid unit
        if end_idx <= start_idx:
            end_idx = start_idx + 1

        # WI: Note Clipping Prevention (already handled by above)

        duration_idx = end_idx - start_idx

        # Map velocity
        # norm_rms = e.velocity / max_rms (if max_rms > 0)
        # linear map: 20 + norm_rms * (105 - 20)
        norm_rms = e.velocity / max_rms if max_rms > 0 else 0
        midi_vel = int(20 + norm_rms * 85)
        midi_vel = max(20, min(105, midi_vel))
        e.velocity = midi_vel / 127.0 # Store as 0-1 for internal, music21 takes 0-127 usually in volume.velocity

        # Convert to beats
        start_beat_global = start_idx * 0.25
        duration_beats = duration_idx * 0.25

        e.duration_beats = duration_beats
        e.measure = int(start_beat_global // beats_per_measure) + 1
        e.beat = (start_beat_global % beats_per_measure) + 1.0

        # Store temporary structure for merging
        # (start_beat, duration_beat, event, midi_vel)
        quantized_notes_temp.append({
            "start": start_beat_global,
            "end": start_beat_global + duration_beats,
            "duration": duration_beats,
            "pitch": e.midi_note,
            "velocity": midi_vel,
            "event": e
        })

    # 3. Gap Merge Rule (WI: "Merge notes if same pitch, separation < 1/32 note")
    # 1/32 note = 0.125 beats.

    merged_notes = []
    if quantized_notes_temp:
        curr = quantized_notes_temp[0]
        for next_n in quantized_notes_temp[1:]:
            gap = next_n["start"] - curr["end"]

            if next_n["pitch"] == curr["pitch"] and gap < 0.125 and gap >= -0.01:
                # Merge
                new_end = next_n["end"]
                new_dur = new_end - curr["start"]
                curr["end"] = new_end
                curr["duration"] = new_dur
                # Average velocity?
                curr["velocity"] = int((curr["velocity"] + next_n["velocity"]) / 2)
                # Update event object (optional, but good for consistency)
                curr["event"].duration_beats = new_dur
                curr["event"].velocity = curr["velocity"] / 127.0
            else:
                merged_notes.append(curr)
                curr = next_n
        merged_notes.append(curr)

    # 4. Build Music21 Score
    s = stream.Score()
    p = stream.Part()

    m1 = stream.Measure()
    m1.number = 1
    m1.timeSignature = meter.TimeSignature(ts_str)
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

    for item in merged_notes:
        n = note.Note(item["pitch"])
        n.quarterLength = item["duration"]
        n.volume.velocity = item["velocity"]

        # Insert at offset
        p.insert(item["start"], n)

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

    if analysis_data.vexflow_layout:
        try:
             if s.hasMeasures():
                part = s.parts[0]
                measures = part.getElementsByClass(stream.Measure)
                analysis_data.vexflow_layout.measures = [{"index": m.number} for m in measures]
        except:
            analysis_data.vexflow_layout.measures = []

    return musicxml_str

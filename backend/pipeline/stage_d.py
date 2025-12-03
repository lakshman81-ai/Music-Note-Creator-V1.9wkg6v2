from __future__ import annotations

from typing import List

from music21 import (
    stream,
    note as m21note,
    meter,
    key as m21key,
    tempo as m21tempo,
    metadata as m21meta,
)

from .models import NoteEvent, AnalysisData, VexflowLayout


def _quantize_notes_to_beats(
    events: List[NoteEvent],
    tempo_bpm: float,
    time_signature: str = "4/4",
) -> None:
    """
    Fill note.measure, note.beat, note.duration_beats.
    Very simple quantizer: snaps to nearest 1/16 note grid.
    """
    quarter_sec = 60.0 / tempo_bpm
    sixteenth_sec = quarter_sec / 4.0

    beats_per_measure = int(time_signature.split("/")[0])  # assume x/4

    for e in events:
        # Quantize start & end in units of sixteenth notes
        n_start = round(e.start_sec / sixteenth_sec)
        n_end = round(e.end_sec / sixteenth_sec)
        if n_end <= n_start:
            n_end = n_start + 1  # at least one sixteenth

        # Convert to beats (beat = quarter note)
        start_beats = n_start / 4.0
        end_beats = n_end / 4.0
        duration_beats = end_beats - start_beats

        measure = int(start_beats // beats_per_measure) + 1
        beat_in_measure = (start_beats % beats_per_measure) + 1.0

        e.measure = measure
        e.beat = beat_in_measure
        e.duration_beats = duration_beats


def _duration_beats_to_quarter_length(duration_beats: float) -> float:
    """
    Map beats (assuming beat = quarter note) to music21 quarterLength.
    """
    # Beat == quarter note under 4/4 assumption.
    return float(duration_beats)


def quantize_and_render(events: List[NoteEvent], analysis_data: AnalysisData) -> str:
    """
    Stage D: Quantization & MusicXML rendering.
    """
    meta = analysis_data.meta
    tempo_bpm = meta.tempo_bpm or 120.0
    time_sig = meta.time_signature or "4/4"

    # 1. Quantize events
    _quantize_notes_to_beats(events, tempo_bpm=tempo_bpm, time_signature=time_sig)

    # 2. Build music21 Score
    s = stream.Score()
    s.insert(0, m21meta.Metadata())
    s.metadata.title = "Transcription"
    s.metadata.composer = "Music-Note-Creator"

    part = stream.Part()
    part.id = "P1"

    # Time signature & tempo
    ts_num, ts_den = time_sig.split("/")
    part.append(m21tempo.MetronomeMark(number=tempo_bpm))
    part.append(meter.TimeSignature(f"{ts_num}/{ts_den}"))

    # Optional key signature
    if meta.detected_key:
        part.append(m21key.Key(meta.detected_key))

    # Insert notes in chronological order
    for e in sorted(events, key=lambda x: (x.measure or 0, x.beat or 0, x.start_sec)):
        n = m21note.Note(e.midi_note)
        ql = _duration_beats_to_quarter_length(e.duration_beats or 1.0)
        n.quarterLength = max(ql, 0.25)  # at least a 16th
        part.append(n)

    s.insert(0, part)

    # 3. Vexflow layout summary (for frontend staff layout)
    measures = []
    for m in part.getElementsByClass('Measure'):
        measures.append(
            {
                "number": int(m.number),
                "width": 0,  # front-end decides actual width
            }
        )
    analysis_data.vexflow_layout = VexflowLayout(measures=measures)

    # 4. Export to MusicXML string
    # Depending on your music21 version, write() may return a path or string.
    # Here we route via a temporary file to be safe.
    from io import BytesIO
    from music21.musicxml import m21ToXml

    xml_out = BytesIO()
    conv = m21ToXml.GeneralObjectExporter(s)
    conv.write(xml_out)
    xml_str = xml_out.getvalue().decode("utf-8")

    return xml_str

from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
import enum

@dataclass
class MetaData:
    tuning_offset: float = 0.0
    detected_key: str = "C"
    lufs: float = -14.0
    processing_mode: str = "polyphonic"
    snr: float = 0.0
    window_size: int = 2048

@dataclass
class AlternativePitch:
    midi: int
    confidence: float

@dataclass
class NoteEvent:
    id: str
    midi_note: int
    start_sec: float
    end_sec: float
    start_beat: float
    duration_beat: float
    confidence: float
    type: str  # "normal", "grace"
    voice: int
    alternatives: List[AlternativePitch] = field(default_factory=list)
    spec_thumb: Optional[str] = None

    # Internal usage fields (not necessarily in final JSON unless requested)
    amplitude: float = 0.0
    dynamic: str = "mf"  # p, mf, f
    is_tuplet: bool = False
    tuplet_info: Optional[str] = None # e.g. "3/2"

@dataclass
class ChordEvent:
    beat: float
    symbol: str # e.g., "Am7"
    timestamp: float = 0.0

@dataclass
class VexflowLayout:
    # This matches the "vexflow_layout": { ... } placeholder.
    # We will populate it with logical measure data.
    measures: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class AnalysisData:
    schema_version: str = "1.0.0"
    meta: MetaData = field(default_factory=MetaData)
    events: List[NoteEvent] = field(default_factory=list)
    chords: List[ChordEvent] = field(default_factory=list)
    vexflow_layout: VexflowLayout = field(default_factory=VexflowLayout)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "meta": self.meta.__dict__,
            "events": [
                {
                    "id": e.id,
                    "midi_note": e.midi_note,
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "start_beat": e.start_beat,
                    "duration_beat": e.duration_beat,
                    "confidence": e.confidence,
                    "type": e.type,
                    "voice": e.voice,
                    "alternatives": [a.__dict__ for a in e.alternatives],
                    "spec_thumb": e.spec_thumb
                } for e in self.events
            ],
            "chords": [
                {"beat": c.beat, "symbol": c.symbol} for c in self.chords
            ],
            "vexflow_layout": self.vexflow_layout.measures
        }

@dataclass
class TranscriptionResult:
    musicxml: str
    analysis_data: AnalysisData

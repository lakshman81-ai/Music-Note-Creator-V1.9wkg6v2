from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class StageAConfig:
    target_sample_rate: int = 44100
    channel_handling: str = "mono_sum" # "mono_sum", "left_only", "right_only"
    dc_offset_removal: bool = True
    transient_pre_emphasis: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "alpha": 0.97})
    high_pass_filter_cutoff: Dict[str, Any] = field(default_factory=lambda: {"value": 55.0})
    high_pass_filter_order: Dict[str, Any] = field(default_factory=lambda: {"value": 4})
    silence_trimming: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "top_db": 50})
    loudness_normalization: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "target_lufs": -23.0})
    noise_floor_estimation: Dict[str, Any] = field(default_factory=lambda: {"percentile": 30})
    bpm_detection: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})

@dataclass
class StageBConfig:
    separation: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "model": "htdemucs"})
    confidence_voicing_threshold: float = 0.75
    confidence_priority_floor: float = 0.5
    pitch_disagreement_cents: float = 70.0
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "swiftf0": 1.0,
        "yin": 0.8,
        "sacf": 0.8,
        "cqt": 0.6,
        "rmvpe": 0.9,
        "crepe": 0.9
    })
    polyphonic_peeling: Dict[str, Any] = field(default_factory=lambda: {"max_layers": 8, "mask_width": 0.03})
    detectors: Dict[str, Any] = field(default_factory=lambda: {
        "rmvpe": {"enabled": False, "fmin": 50.0, "fmax": 1200.0, "hop_length": 160, "silence_threshold": 0.04},
        "crepe": {"enabled": False, "model_capacity": "full", "fmin": 190.0, "fmax": 3500.0, "use_viterbi": False},
        "swiftf0": {"enabled": True},
        "yin": {"enabled": True},
        "sacf": {"enabled": True},
        "cqt": {"enabled": True}
    })

@dataclass
class StageCConfig:
    min_note_duration_ms: float = 30.0
    frame_stability: Dict[str, Any] = field(default_factory=lambda: {"stable_frames_required": 2})
    pitch_tolerance_cents: float = 50.0
    gap_filling: Dict[str, Any] = field(default_factory=lambda: {"max_gap_ms": 100.0})
    velocity_map: Dict[str, float] = field(default_factory=lambda: {
        "min_db": -40.0, "max_db": -4.0, "min_vel": 20.0, "max_vel": 105.0
    })
    polyphony_filter: Dict[str, str] = field(default_factory=lambda: {"mode": "skyline_top_voice"})

@dataclass
class StageDConfig:
    divisions_per_quarter: int = 24 # Standard MusicXML
    staff_split_point: Dict[str, Any] = field(default_factory=lambda: {"pitch": 60}) # C4
    staccato_marking: Dict[str, Any] = field(default_factory=lambda: {"threshold_beats": 0.25})
    glissando_threshold_general: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "min_semitones": 2.0, "max_time_ms": 500})
    glissando_handling_piano: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})

@dataclass
class InstrumentProfile:
    instrument: str
    recommended_algo: str
    fmin: float
    fmax: float
    special: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    stage_c: StageCConfig = field(default_factory=StageCConfig)
    stage_d: StageDConfig = field(default_factory=StageDConfig)
    instrument_profiles: List[InstrumentProfile] = field(default_factory=list)

    def get_profile(self, instrument_name: str) -> Optional[InstrumentProfile]:
        for p in self.instrument_profiles:
            if p.instrument == instrument_name:
                return p
        return None

# --- Default Configuration Instance ---

# Instrument Profiles from WI
_profiles = [
    InstrumentProfile("piano_61key", "swiftf0", 60.0, 2200.0, {"viterbi": False, "ensemble_smoothing_frames": 3}),
    InstrumentProfile("vocals", "rmvpe", 50.0, 1200.0, {"viterbi": True, "silence_threshold": 0.04}),
    InstrumentProfile("violin", "crepe", 190.0, 3500.0, {"viterbi": True}),
    InstrumentProfile("bass_guitar", "yin", 30.0, 400.0, {"frame_length": 8192}),
    InstrumentProfile("cello", "rmvpe", 65.0, 880.0, {"viterbi": True}),
    InstrumentProfile("flute", "crepe", 261.0, 3349.0, {"small_window": True}),
    InstrumentProfile("acoustic_guitar", "swiftf0", 82.0, 880.0, {"smoothing_frames": 3}),
    InstrumentProfile("electric_guitar_clean", "yin", 80.0, 1200.0, {"threshold": 0.05}),
    InstrumentProfile("drums_percussive", "none", 0.0, 0.0, {"ignore_pitch": True, "high_conf_threshold": 0.15}),
]

PIANO_61KEY_CONFIG = PipelineConfig(
    instrument_profiles=_profiles
)

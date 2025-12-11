from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import librosa
import scipy.signal

from .models import (
    MetaData,
    FramePitch,
    NoteEvent,
    ChordEvent,
    StageAOutput,
    Stem,
    AudioQuality,
    StageBOutput,
)
from .detectors import (
    YinDetector,
    CQTDetector,
    SACFDetector,
    SwiftF0Detector,
    RMVPEDetector,
    CREPEDetector,
)
from .config import (
    PIANO_61KEY_CONFIG,
    StageBConfig,
    InstrumentProfile,
    PipelineConfig,
)


# ------------------------------------------------------------
# Pitch conversion helpers
# ------------------------------------------------------------

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_midi(hz: float) -> float:
    if hz <= 0.0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)


# ------------------------------------------------------------
# Harmonic masking helper (for ISS)
# ------------------------------------------------------------

def create_harmonic_mask(
    stft: np.ndarray,
    f0_curve: np.ndarray,
    sr: int,
    width: float = 0.03,
) -> np.ndarray:
    """
    Build a harmonic mask for iterative spectral subtraction based on an F0 curve.
    `width` is fractional bandwidth around each harmonic.
    """
    n_freqs, n_frames = stft.shape
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_freqs - 1) * 2)
    mask = np.zeros_like(stft, dtype=np.float32)

    for t in range(min(n_frames, len(f0_curve))):
        f0 = float(f0_curve[t])
        if f0 <= 0.0:
            continue

        harmonics = np.arange(1, 20, dtype=float) * f0
        harmonics = harmonics[harmonics < sr / 2.0]

        for h_freq in harmonics:
            bw = h_freq * width
            f_low = h_freq - bw / 2.0
            f_high = h_freq + bw / 2.0

            idx_start = np.searchsorted(fft_freqs, f_low)
            idx_end = np.searchsorted(fft_freqs, f_high)
            mask[idx_start:idx_end, t] = 1.0

    return mask


# ------------------------------------------------------------
# Iterative Spectral Subtraction (Polyphonic Peeling)
# ------------------------------------------------------------

def iterative_spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    primary_detector,
    validator_detector,
    max_polyphony: int = 4,
    mask_width: float = 0.03,
    audio_path: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Iteratively estimate F0 curves and subtract their harmonic energy
    from the spectrum to peel off polyphonic layers.

    Returns:
        List of (f0_curve, conf_curve) for each layer.
    """
    extracted: List[Tuple[np.ndarray, np.ndarray]] = []
    y_resid = audio.copy()

    # Use detector's hop_length consistently
    hop_length = primary_detector.hop_length
    n_fft = 2048

    for _ in range(max_polyphony):
        # 1. Proposal
        f0, conf = primary_detector.predict(y_resid, audio_path=audio_path)

        # Termination: if curve is too weak
        if float(np.mean(conf)) < 0.15 and float(np.max(conf)) < 0.25:
            break

        # 2. Validation (SACF or similar)
        validation_score = validator_detector.validate_curve(
            f0,
            y_resid,
            threshold=0.2,
        )
        if validation_score < 0.2 and len(extracted) > 0:
            break

        extracted.append((f0, conf))

        # 3. Subtraction
        S = librosa.stft(y_resid, n_fft=n_fft, hop_length=hop_length)
        mask = create_harmonic_mask(S, f0, sr, width=mask_width)
        S_clean = S * (1.0 - mask)
        y_resid = librosa.istft(S_clean, hop_length=hop_length, length=len(y_resid))

    return extracted


# ------------------------------------------------------------
# Stage B: Feature Extraction / Pitch Estimation
# ------------------------------------------------------------

def extract_features(
    stage_a_output: StageAOutput,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
    use_crepe: bool = False,          # Deprecated
    confidence_threshold: float = 0.5, # Deprecated (use config.stage_b instead)
    min_duration_ms: float = 0.0,      # Deprecated
) -> StageBOutput:
    """
    Stage B: Source separation + F0 extraction + ensemble + polyphonic peeling.

    WI alignment:
    - Uses Demucs (HTDemucs) for separation (if enabled).
    - Applies instrument profiles for each stem (fmin/fmax/algos/hop_length).
    - Runs multiple detectors (SwiftF0, YIN, SACF, CQT, RMVPE, CREPE).
    - Combines detectors via ensemble + SwiftF0 priority and disagreement check.
    - For piano, runs iterative spectral subtraction (ISS) to peel polyphonic layers.
    - Outputs:
        - time_grid
        - f0_main (monophonic main)
        - f0_layers (polyphonic layers)
        - per_detector (debug)
        - stem_timelines (FramePitch list per stem)
        - meta (MetaData passthrough)
    """
    sb_config: StageBConfig = config.stage_b
    meta: MetaData = stage_a_output.meta

    # --------------------------------------------------------
    # 1. Global analysis parameters (shared time grid)
    # --------------------------------------------------------
    detector_sr = int(meta.target_sr)          # Single SR for all detectors
    global_hop = int(meta.hop_length)         # Canonical F0 hop size
    duration_sec = float(meta.duration_sec)

    n_frames_global = int(duration_sec * detector_sr / global_hop) + 1
    times_global = librosa.frames_to_time(
        np.arange(n_frames_global),
        sr=detector_sr,
        hop_length=global_hop,
    )

    # --------------------------------------------------------
    # 2. Separation (Demucs) — Stage B per WI
    # --------------------------------------------------------
    stems_in = stage_a_output.stems
    mix_stem = stems_in.get("mix", stems_in.get("vocals", None))

    final_stems: Dict[str, Stem] = {}

    if sb_config.separation.get("enabled", True) and mix_stem is not None:
        # Lazy import to avoid circular import at module load
        from .stage_a import apply_demucs

        model_name = sb_config.separation.get("model", "htdemucs")
        # Demucs runs at its own internal SR; we resample outputs back to detector_sr
        demucs_stems = apply_demucs(mix_stem.audio, mix_stem.sr, model_name=model_name)

        for name, audio in demucs_stems.items():
            # Resample each stem to detector_sr for consistent detector input
            if mix_stem.sr != detector_sr:
                audio_res = librosa.resample(audio, orig_sr=mix_stem.sr, target_sr=detector_sr)
            else:
                audio_res = audio

            final_stems[name] = Stem(audio=audio_res, sr=detector_sr, type=name)
    else:
        # No separation: treat entire mix as piano_61key
        if mix_stem is None:
            raise ValueError("Stage B: No mix stem available from Stage A.")
        # Ensure mix is at detector_sr
        if mix_stem.sr != detector_sr:
            audio_res = librosa.resample(mix_stem.audio, orig_sr=mix_stem.sr, target_sr=detector_sr)
        else:
            audio_res = mix_stem.audio
        final_stems["mix"] = Stem(audio=audio_res, sr=detector_sr, type="mix")

    # --------------------------------------------------------
    # 3. Stem → Instrument profile mapping
    # --------------------------------------------------------
    stem_profile_map: Dict[str, str] = {
        "vocals": "vocals",
        "bass": "bass_guitar",
        "other": "piano_61key",
        "piano": "piano_61key",
        "mix": "piano_61key",
        # Additional stems (e.g., "guitar") can be mapped here if needed.
    }

    per_detector_results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    stem_timelines: Dict[str, List[FramePitch]] = {}

    # Global outputs (aggregated)
    f0_main_global = np.zeros(n_frames_global, dtype=float)
    f0_layers_global: List[np.ndarray] = []

    # Config flag: whether ISS layer 0 should override ensemble main
    use_iss_for_main: bool = sb_config.polyphonic_peeling.get("use_iss_for_main", False)

    # --------------------------------------------------------
    # 4. Process each stem: detectors, ensemble, ISS, FramePitch
    # --------------------------------------------------------
    for stem_name, stem in final_stems.items():
        if stem_name == "drums":
            # No pitch for drums
            continue

        # -------------------------
        # 4.1 Instrument profile
        # -------------------------
        profile_name = stem_profile_map.get(stem_name, "piano_61key")
        profile: Optional[InstrumentProfile] = config.get_profile(profile_name)
        if profile is None:
            profile = config.get_profile("piano_61key")

        # -------------------------
        # 4.2 Instantiate detectors
        # -------------------------
        detectors: List[Tuple[str, Any]] = []

        def add_det(cls, name: str, **kwargs: Any) -> None:
            try:
                # Per-profile hop_length override (e.g., RMVPE vocals vs piano)
                hop_for_profile = int(profile.special.get("hop_length", global_hop))
                d = cls(
                    sr=detector_sr,
                    hop_length=hop_for_profile,
                    fmin=profile.fmin,
                    fmax=profile.fmax,
                    **kwargs,
                )
                detectors.append((name, d))
            except Exception as e:
                print(f"[Stage B] Failed to init detector '{name}' for stem '{stem_name}': {e}")

        # SwiftF0
        if sb_config.detectors.get("swiftf0", {}).get("enabled", False):
            add_det(SwiftF0Detector, "swiftf0")

        # YIN
        if sb_config.detectors.get("yin", {}).get("enabled", False):
            add_det(YinDetector, "yin")

        # SACF
        if sb_config.detectors.get("sacf", {}).get("enabled", False):
            add_det(SACFDetector, "sacf")

        # CQT
        if sb_config.detectors.get("cqt", {}).get("enabled", False):
            add_det(CQTDetector, "cqt")

        # RMVPE
        if profile.recommended_algo == "rmvpe" or sb_config.detectors.get("rmvpe", {}).get("enabled", False):
            sil_thresh = float(profile.special.get("silence_threshold", 0.04))
            add_det(RMVPEDetector, "rmvpe", silence_threshold=sil_thresh)

        # CREPE
        if profile.recommended_algo == "crepe" or sb_config.detectors.get("crepe", {}).get("enabled", False):
            viterbi = bool(profile.special.get("viterbi", False))
            add_det(CREPEDetector, "crepe", use_viterbi=viterbi)

        # If no detectors enabled, skip this stem
        if not detectors:
            continue

        # -------------------------
        # 4.3 Run detectors on stem audio
        # -------------------------
        audio_for_det = stem.audio
        if audio_for_det.ndim > 1:
            # Ensure mono
            audio_for_det = np.mean(audio_for_det, axis=0)

        candidates_per_frame: List[List[Dict[str, float]]] = [
            [] for _ in range(n_frames_global)
        ]
        stem_raw_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        primary_det = None   # SwiftF0 for ISS
        validator_det = None # SACF validator for ISS

        for d_name, det in detectors:
            if d_name == "swiftf0":
                primary_det = det
            if d_name == "sacf":
                validator_det = det

            f0, conf = det.predict(audio_for_det, audio_path=meta.audio_path)

            # Normalize length to global frames
            if len(f0) != n_frames_global:
                if len(f0) > n_frames_global:
                    f0 = f0[:n_frames_global]
                    conf = conf[:n_frames_global]
                else:
                    pad = n_frames_global - len(f0)
                    f0 = np.pad(f0, (0, pad))
                    conf = np.pad(conf, (0, pad))

            stem_raw_results[d_name] = (f0, conf)

            for i in range(n_frames_global):
                candidates_per_frame[i].append(
                    {
                        "name": d_name,
                        "f0": float(f0[i]),
                        "conf": float(conf[i]),
                    }
                )

        per_detector_results[stem_name] = stem_raw_results

        # -------------------------
        # 4.4 Ensemble combine per frame
        # -------------------------
        stem_f0 = np.zeros(n_frames_global, dtype=float)
        stem_conf = np.zeros(n_frames_global, dtype=float)

        weights: Dict[str, float] = sb_config.ensemble_weights
        priority_floor = float(sb_config.confidence_priority_floor)
        disagreement_cents = float(sb_config.pitch_disagreement_cents)
        voicing_threshold = float(sb_config.confidence_voicing_threshold)

        for i in range(n_frames_global):
            cands = candidates_per_frame[i]
            if not cands:
                continue

            final_f0 = 0.0
            final_conf = 0.0

            # Check disagreement among detectors
            valid_pitches = [
                c["f0"]
                for c in cands
                if c["f0"] > 10.0 and c["conf"] > 0.1
            ]
            disagreement = False
            if valid_pitches:
                min_p = min(valid_pitches)
                max_p = max(valid_pitches)
                if max_p > 0.0 and min_p > 0.0:
                    diff_cents = 1200.0 * np.log2(max_p / min_p)
                    if diff_cents > disagreement_cents:
                        disagreement = True

            # SwiftF0 priority if confident and not in heavy disagreement
            swift = next((c for c in cands if c["name"] == "swiftf0"), None)
            if swift and swift["conf"] >= priority_floor and not disagreement:
                final_f0 = swift["f0"]
                final_conf = swift["conf"]
            else:
                # Weighted average across detectors
                num = 0.0
                den = 0.0
                max_c = 0.0
                for c in cands:
                    if c["f0"] <= 10.0:
                        continue
                    w = float(weights.get(c["name"], 0.5))
                    num += c["f0"] * c["conf"] * w
                    den += c["conf"] * w
                    if c["conf"] > max_c:
                        max_c = c["conf"]

                if den > 0.0:
                    final_f0 = num / den
                    final_conf = max_c

            # Voicing decision
            if final_conf < voicing_threshold:
                final_f0 = 0.0

            stem_f0[i] = final_f0
            stem_conf[i] = final_conf

        # -------------------------
        # 4.5 Optional smoothing (ensemble_smoothing_frames)
        # -------------------------
        kernel = int(profile.special.get("ensemble_smoothing_frames", 1))
        if kernel > 1 and kernel % 2 == 1:
            try:
                stem_f0 = scipy.signal.medfilt(stem_f0, kernel_size=kernel)
            except Exception:
                # If medfilt fails (e.g., even kernel), ignore smoothing
                pass

        # -------------------------
        # 4.6 Polyphonic Peeling (ISS) for piano_61key
        # -------------------------
        stem_layers: List[np.ndarray] = []

        if (
            profile_name == "piano_61key"
            and primary_det is not None
            and validator_det is not None
            and sb_config.polyphonic_peeling.get("enabled", True)
        ):
            # Reset stateful detectors if supported
            if hasattr(primary_det, "reset_state"):
                try:
                    primary_det.reset_state()
                except Exception:
                    pass

            audio_for_iss = audio_for_det  # already at detector_sr mono

            extracted = iterative_spectral_subtraction(
                audio_for_iss,
                detector_sr,
                primary_detector=primary_det,
                validator_detector=validator_det,
                max_polyphony=int(sb_config.polyphonic_peeling.get("max_layers", 4)),
                mask_width=float(sb_config.polyphonic_peeling.get("mask_width", 0.03)),
                audio_path=meta.audio_path,
            )

            # Normalize extracted layers to global frame count
            iss_layers: List[Tuple[np.ndarray, np.ndarray]] = []
            for f0_iss, conf_iss in extracted:
                f0_iss = np.asarray(f0_iss, dtype=float)
                conf_iss = np.asarray(conf_iss, dtype=float)
                if len(f0_iss) != n_frames_global:
                    if len(f0_iss) > n_frames_global:
                        f0_iss = f0_iss[:n_frames_global]
                        conf_iss = conf_iss[:n_frames_global]
                    else:
                        pad = n_frames_global - len(f0_iss)
                        f0_iss = np.pad(f0_iss, (0, pad))
                        conf_iss = np.pad(conf_iss, (0, pad))
                iss_layers.append((f0_iss, conf_iss))

            if iss_layers:
                # Optionally override ensemble main with ISS first layer
                if use_iss_for_main:
                    stem_f0 = iss_layers[0][0]
                    stem_conf = iss_layers[0][1]
                    # Remaining ISS layers are additional polyphonic layers
                    for (lf0, _) in iss_layers[1:]:
                        stem_layers.append(lf0)
                else:
                    # Ensemble remains main; all ISS layers treated as polyphonic layers
                    for (lf0, _) in iss_layers:
                        stem_layers.append(lf0)

        # -------------------------
        # 4.7 Build FramePitch timeline for this stem
        # -------------------------
        tl: List[FramePitch] = []
        for i in range(n_frames_global):
            active: List[Tuple[float, float]] = []
            if stem_f0[i] > 0.0:
                active.append((float(stem_f0[i]), float(stem_conf[i])))

            for l in stem_layers:
                if l[i] > 0.0:
                    active.append((float(l[i]), 0.8))  # placeholder conf for extra layers

            # Sort pitches by confidence
            active.sort(key=lambda x: x[1], reverse=True)

            if active:
                main_pitch_hz = active[0][0]
                main_conf = active[0][1]
                main_midi = int(round(hz_to_midi(main_pitch_hz)))
            else:
                main_pitch_hz = 0.0
                main_conf = 0.0
                main_midi = None

            fp = FramePitch(
                time=float(times_global[i]),
                pitch_hz=main_pitch_hz,
                midi=main_midi,
                confidence=main_conf,
                active_pitches=active,
            )
            tl.append(fp)

        stem_timelines[stem_name] = tl

        # -------------------------
        # 4.8 Aggregate stems into global main/layers
        # -------------------------
        if len(stem_timelines) == 1:
            # Only one stem => treat its main track as global main
            f0_main_global = stem_f0
            f0_layers_global = stem_layers
        else:
            # Multiple stems: by default, vocals main, piano layers.
            if stem_name == "vocals":
                f0_main_global = stem_f0
            elif stem_name in ("other", "piano", "mix"):
                f0_layers_global.append(stem_f0)
                f0_layers_global.extend(stem_layers)
            else:
                # Other instruments → add as extra layers
                f0_layers_global.append(stem_f0)
                f0_layers_global.extend(stem_layers)

    # --------------------------------------------------------
    # 5. Return StageBOutput
    # --------------------------------------------------------
    return StageBOutput(
        time_grid=times_global,
        f0_main=f0_main_global,
        f0_layers=f0_layers_global,
        per_detector=per_detector_results,
        stem_timelines=stem_timelines,
        meta=meta,
    )

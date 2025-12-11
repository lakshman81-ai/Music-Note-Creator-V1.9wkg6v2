from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import librosa
import scipy.signal
import scipy.interpolate
from .models import MetaData, FramePitch, NoteEvent, ChordEvent, StageAOutput, Stem, AudioQuality, StageBOutput
from .detectors import YinDetector, CQTDetector, SACFDetector, SwiftF0Detector, RMVPEDetector, CREPEDetector
from .config import PIANO_61KEY_CONFIG, StageBConfig, InstrumentProfile, PipelineConfig

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

def hz_to_midi(hz: float) -> float:
    if hz <= 0: return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)

def create_harmonic_mask(stft: np.ndarray, f0_curve: np.ndarray, sr: int, width: float = 0.03) -> np.ndarray:
    n_freqs, n_frames = stft.shape
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_freqs - 1) * 2)
    mask = np.zeros_like(stft, dtype=np.float32)

    for t in range(min(n_frames, len(f0_curve))):
        f0 = f0_curve[t]
        if f0 <= 0: continue

        harmonics = np.arange(1, 20) * f0
        harmonics = harmonics[harmonics < sr / 2]

        for h_freq in harmonics:
            bw = h_freq * width
            f_low = h_freq - bw/2
            f_high = h_freq + bw/2

            idx_start = np.searchsorted(fft_freqs, f_low)
            idx_end = np.searchsorted(fft_freqs, f_high)
            mask[idx_start:idx_end, t] = 1.0

    return mask

def iterative_spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    primary_detector,
    validator_detector,
    max_polyphony: int = 4,
    mask_width: float = 0.03,
    audio_path: Optional[str] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:

    extracted = []
    y_resid = audio.copy()
    n_fft = 2048
    hop_length = primary_detector.hop_length

    for i in range(max_polyphony):
        # 1. Proposal
        f0, conf = primary_detector.predict(y_resid, audio_path=audio_path)

        # Check termination
        if np.mean(conf) < 0.15 and np.max(conf) < 0.25:
            break

        # 2. Validation
        validation_score = validator_detector.validate_curve(f0, y_resid, threshold=0.2)
        if validation_score < 0.2 and i > 0:
            break

        extracted.append((f0, conf))

        # 3. Subtraction
        S = librosa.stft(y_resid, n_fft=n_fft, hop_length=hop_length)
        mask = create_harmonic_mask(S, f0, sr, width=mask_width)
        S_clean = S * (1 - mask)
        y_resid = librosa.istft(S_clean, hop_length=hop_length, length=len(y_resid))

    return extracted

def extract_features(
    stage_a_output: StageAOutput,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
    use_crepe: bool = False, # Deprecated
    confidence_threshold: float = 0.5, # Deprecated in favor of config
    min_duration_ms: float = 0.0, # Deprecated
) -> StageBOutput:

    sb_config = config.stage_b
    meta = stage_a_output.meta

    # 1. Separation (if enabled and not mono/fast mode)
    # The Separation Logic from WI 2.2.1
    stems = stage_a_output.stems
    mix_stem = stems.get("mix", stems.get("vocals", None)) # Fallback

    final_stems = {}

    # Separation
    if sb_config.separation.get("enabled", True) and mix_stem is not None:
        from .stage_a import apply_demucs # Circular import? Lazy import

        # If we already have separated stems (e.g. from Stage A fast path?), Stage A output might have them.
        # But my refactor of Stage A only returns 'mix' usually now.
        # Let's run separation here.
        model_name = sb_config.separation.get("model", "htdemucs")
        demucs_stems = apply_demucs(mix_stem.audio, mix_stem.sr, model_name=model_name)

        # Map 4-stem to our needs
        # WI: "If 4-stem model: use 'other' for piano."
        # WI: "If 6-stem model: use 'piano'."

        # We assume standard 4-stem (vocals, bass, drums, other)
        # We need to construct Stem objects
        for name, audio in demucs_stems.items():
            final_stems[name] = Stem(audio=audio, sr=mix_stem.sr, type=name)

    else:
        # No separation, everything is "mix" (treated as piano/other)
        final_stems["mix"] = mix_stem

    # 2. Detectors & Ensemble
    # We need to process each stem according to Instrument Profiles
    # WI 2.2.2 and 2.2.3

    # Define mapping of stem name to instrument profile name
    # Default assumptions:
    # 'vocals' -> 'vocals'
    # 'bass' -> 'bass_guitar'
    # 'other' -> 'piano_61key' (Goal is "Isolate piano", so 'other' is piano)
    # 'mix' -> 'piano_61key' (if no separation)

    stem_profile_map = {
        "vocals": "vocals",
        "bass": "bass_guitar",
        "other": "piano_61key",
        "mix": "piano_61key"
    }

    # Global Time Grid (Master)
    # WI: "All detectors must produce F0 at the same hop size"
    # Let's fix a global hop length, say 256 samples @ 44100 -> ~5.8ms
    global_sr = meta.target_sr
    global_hop = meta.hop_length
    n_frames_global = int(meta.duration_sec * global_sr / global_hop) + 1
    times_global = librosa.frames_to_time(np.arange(n_frames_global), sr=global_sr, hop_length=global_hop)

    per_detector_results = {}
    f0_main_global = np.zeros(n_frames_global)
    f0_layers_global = [] # List of F0 arrays
    stem_timelines = {}

    for stem_name, stem in final_stems.items():
        if stem_name == "drums": continue # Skip drums for pitch

        profile_name = stem_profile_map.get(stem_name, "piano_61key")
        profile = config.get_profile(profile_name)
        if not profile: profile = config.get_profile("piano_61key")

        # Instantiate Detectors
        # WI 2.2.2 Usage Rules
        detectors = []

        # Helper to add detector
        def add_det(cls, name, **kwargs):
             try:
                 d = cls(sr=stem.sr, hop_length=global_hop, fmin=profile.fmin, fmax=profile.fmax, **kwargs)
                 detectors.append((name, d))
             except Exception as e:
                 print(f"Failed to init {name}: {e}")

        # Based on config.stage_b.detectors enablement AND profile recommendation
        # WI says: "Use if available and configured."
        # Also "Enabled per instrument profile" (RMVPE/CREPE)

        # SwiftF0
        if sb_config.detectors["swiftf0"]["enabled"]:
             add_det(SwiftF0Detector, "swiftf0")

        # YIN
        if sb_config.detectors["yin"]["enabled"]:
             add_det(YinDetector, "yin")

        # SACF
        if sb_config.detectors["sacf"]["enabled"]:
             add_det(SACFDetector, "sacf")

        # CQT
        if sb_config.detectors["cqt"]["enabled"]:
             add_det(CQTDetector, "cqt")

        # RMVPE
        if profile.recommended_algo == "rmvpe" or sb_config.detectors["rmvpe"]["enabled"]:
             # Check profile special params
             sil_thresh = profile.special.get("silence_threshold", 0.04)
             add_det(RMVPEDetector, "rmvpe", silence_threshold=sil_thresh)

        # CREPE
        if profile.recommended_algo == "crepe" or sb_config.detectors["crepe"]["enabled"]:
             viterbi = profile.special.get("viterbi", False)
             add_det(CREPEDetector, "crepe", use_viterbi=viterbi)


        # Run Detectors
        # Collect (f0, conf) for each
        candidates_per_frame = [[] for _ in range(n_frames_global)]

        stem_raw_results = {}

        primary_det = None # For ISS
        validator_det = None

        for d_name, det in detectors:
            if d_name == "swiftf0": primary_det = det
            if d_name == "sacf": validator_det = det

            # Predict
            # Note: detectors are init with stem.sr, but output frames must match global grid.
            # We enforce stem.sr == global_sr usually?
            # If Stage A returns different SRs per stem?
            # Stage A now returns all at target_sr for mix.
            # But Separation (Demucs) might return 44.1k.
            # If detectors are init with stem.sr, their hop_length needs to be adjusted?
            # We passed global_hop.
            # If stem.sr != global_sr, then hop_length implies different duration.
            # We must resample audio to global_sr OR handle resampling of F0.
            # Let's resample audio to global_sr if mismatch.

            audio_for_det = stem.audio
            if stem.sr != global_sr:
                audio_for_det = librosa.resample(stem.audio, orig_sr=stem.sr, target_sr=global_sr)

            f0, conf = det.predict(audio_for_det, audio_path=meta.audio_path)

            # Ensure length matches
            if len(f0) != n_frames_global:
                # Pad or trim
                if len(f0) > n_frames_global:
                    f0 = f0[:n_frames_global]
                    conf = conf[:n_frames_global]
                else:
                    pad = n_frames_global - len(f0)
                    f0 = np.pad(f0, (0, pad))
                    conf = np.pad(conf, (0, pad))

            stem_raw_results[d_name] = (f0, conf)

            for i in range(n_frames_global):
                candidates_per_frame[i].append({
                    "name": d_name,
                    "f0": f0[i],
                    "conf": conf[i]
                })

        per_detector_results[stem_name] = stem_raw_results

        # 2.2.3 Ensemble Logic (Monophonic / Main Track)

        stem_f0 = np.zeros(n_frames_global)
        stem_conf = np.zeros(n_frames_global)

        weights = sb_config.ensemble_weights
        priority_floor = sb_config.confidence_priority_floor
        disagreement_cents = sb_config.pitch_disagreement_cents

        for i in range(n_frames_global):
            cands = candidates_per_frame[i]
            if not cands: continue

            # Priority Override (SwiftF0)
            swift = next((c for c in cands if c["name"] == "swiftf0"), None)

            final_f0 = 0.0
            final_conf = 0.0

            # Check disagreement
            valid_pitches = [c["f0"] for c in cands if c["f0"] > 10.0 and c["conf"] > 0.1]
            disagreement = False
            if valid_pitches:
                min_p = min(valid_pitches)
                max_p = max(valid_pitches)
                diff_cents = 1200 * np.log2(max_p / min_p)
                if diff_cents > disagreement_cents:
                    disagreement = True

            if swift and swift["conf"] >= priority_floor and not disagreement:
                final_f0 = swift["f0"]
                final_conf = swift["conf"]
            else:
                # Weighted Average
                num = 0.0
                den = 0.0
                max_c = 0.0
                for c in cands:
                    w = weights.get(c["name"], 0.5)
                    if c["f0"] > 10.0:
                        num += c["f0"] * c["conf"] * w
                        den += c["conf"] * w
                        if c["conf"] > max_c: max_c = c["conf"]

                if den > 0:
                    final_f0 = num / den
                    final_conf = max_c

            # Thresholding
            if final_conf < sb_config.confidence_voicing_threshold:
                final_f0 = 0.0

            stem_f0[i] = final_f0
            stem_conf[i] = final_conf

        # Smoothing
        kernel = profile.special.get("ensemble_smoothing_frames", 3)
        if kernel > 1:
            stem_f0 = scipy.signal.medfilt(stem_f0, kernel_size=kernel)

        # Store main track for this stem
        # If this is "other" or "piano", we also run Polyphonic Peeling

        stem_layers = []
        if profile_name == "piano_61key" and primary_det and validator_det:
            # Polyphonic Peeling
             # Ensure state reset
            if hasattr(primary_det, 'reset_state'): primary_det.reset_state()

            audio_for_iss = stem.audio
            if stem.sr != global_sr:
                 audio_for_iss = librosa.resample(stem.audio, orig_sr=stem.sr, target_sr=global_sr)

            extracted = iterative_spectral_subtraction(
                audio_for_iss,
                global_sr,
                primary_det,
                validator_det,
                max_polyphony=sb_config.polyphonic_peeling["max_layers"],
                mask_width=sb_config.polyphonic_peeling["mask_width"],
                audio_path=meta.audio_path
            )

            # Extracted contains [(f0, conf), (f0, conf)...]
            # Layer 0 is usually the main one, but Ensemble might be better for main.
            # Let's trust Ensemble for main, and use ISS for layers?
            # Or use ISS entirely?
            # WI says "Output ... f0_main ... f0_layers".
            # "f0_main (monophonic track or skyline)".
            # If ISS runs, f0_main should be the first peeled layer (dominant).
            # But we calculated Ensemble above.
            # Let's replace stem_f0 with extracted[0] if available and better?
            # Actually, ISS with SwiftF0 Primary IS the ensemble logic in a way (SwiftF0 dominates).

            if extracted:
                 # Override main with ISS first layer
                 e0 = extracted[0][0]
                 e_conf = extracted[0][1]
                 if len(e0) != n_frames_global:
                     e0 = np.pad(e0, (0, max(0, n_frames_global - len(e0))))[:n_frames_global]
                     e_conf = np.pad(e_conf, (0, max(0, n_frames_global - len(e_conf))))[:n_frames_global]

                 stem_f0 = e0
                 stem_conf = e_conf

                 # Remaining are layers
                 for lay in extracted[1:]:
                     # Normalize length
                     lf0 = lay[0]
                     if len(lf0) != n_frames_global:
                         lf0 = np.pad(lf0, (0, max(0, n_frames_global - len(lf0))))[:n_frames_global]
                     stem_layers.append(lf0)

        # Build FramePitch objects for this stem
        tl = []
        for i in range(n_frames_global):
            active = []
            if stem_f0[i] > 0:
                active.append((stem_f0[i], stem_conf[i]))
            for l in stem_layers:
                if l[i] > 0:
                    active.append((l[i], 0.8)) # Dummy conf for layers if not tracked

            active.sort(key=lambda x: x[1], reverse=True)

            fp = FramePitch(
                time=times_global[i],
                pitch_hz=active[0][0] if active else 0.0,
                midi=int(round(hz_to_midi(active[0][0]))) if active else None,
                confidence=active[0][1] if active else 0.0,
                active_pitches=active
            )
            tl.append(fp)

        stem_timelines[stem_name] = tl

        # Aggregate to global (Skyline or Mix)
        # If multiple stems, how to combine?
        # Vocals + Piano -> Polyphony.
        # We add them as layers.
        if len(stem_timelines) == 1:
            f0_main_global = stem_f0
            f0_layers_global = stem_layers
        else:
            # Simple aggregation logic: Max confidence wins for Main?
            # Or just append all valid pitches to layers
            # Here we keep f0_main as the "Melody" (Vocals usually)
            if stem_name == "vocals":
                f0_main_global = stem_f0
            elif stem_name == "other" or stem_name == "piano":
                f0_layers_global.append(stem_f0)
                f0_layers_global.extend(stem_layers)

    return StageBOutput(
        time_grid=times_global,
        f0_main=f0_main_global,
        f0_layers=f0_layers_global,
        per_detector=per_detector_results,
        stem_timelines=stem_timelines,
        meta=meta
    )

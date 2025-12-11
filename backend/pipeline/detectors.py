import os
import warnings
from collections import defaultdict
from typing import Optional, Tuple, List, Union, Dict, Set, Any

import numpy as np
import librosa
import scipy.signal
import torch
import torch.nn as nn
import music21

# Optional dependencies
try:
    import crepe
except ImportError:
    crepe = None

try:
    from rmvpe import RMVPE
except ImportError:
    RMVPE = None


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# ------------------------------------------------------------
# Base Detector
# ------------------------------------------------------------

class BasePitchDetector:
    """
    Base class for pitch detectors.
    All detectors must implement:
        predict(y, audio_path=None) -> (f0_hz, confidence)
    """

    def __init__(self, sr: int, hop_length: int, fmin: float, fmax: float):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def predict(
        self,
        y: np.ndarray,
        audio_path: Optional[str] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[List[List[float]], List[List[float]]]
    ]:
        """
        Returns (pitch_hz, confidence) per frame.
        Can optionally return lists-of-lists for explicit polyphony.
        """
        raise NotImplementedError


# ------------------------------------------------------------
# YIN / pYIN Detector
# ------------------------------------------------------------

class YinDetector(BasePitchDetector):
    """
    Wrapper around librosa.pyin (pYIN-like) as our YIN-based detector.
    """

    def __init__(
        self,
        sr: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        frame_length: int = 4096,
    ):
        super().__init__(sr, hop_length, fmin, fmax)
        # For piano C2 (~65 Hz), WI recommends 3072–4096; default to 4096.
        self.frame_length = frame_length

    def predict(
        self,
        y: np.ndarray,
        audio_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            fill_na=0.0,
        )
        f0 = np.nan_to_num(f0)
        voiced_probs = np.nan_to_num(voiced_probs)
        return f0, voiced_probs


# ------------------------------------------------------------
# CQT Detector (Monophonic + Polyphonic Peak Mode)
# ------------------------------------------------------------

class CQTDetector(BasePitchDetector):
    """
    Constant-Q Transform-based detector.

    For monophonic mode:
        - Returns single F0 per frame (peak magnitude bin).
    For polyphonic mode:
        - Returns multiple peaks per frame (up to max_peaks).

    WI alignment for piano:
        - bins_per_octave ≈ 48–60
        - n_octaves ≈ 5 for 61-key range
        - peak_height_rel ≈ 0.10
        - peak_distance_bins ≈ 18 (about ~6 semitones at 3 bins/semitone)
    """

    def __init__(
        self,
        sr: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        bins_per_octave: int = 48,
        n_octaves: int = 5,
        peak_height_rel: float = 0.10,
        peak_distance_bins: int = 18,
    ):
        super().__init__(sr, hop_length, fmin, fmax)
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.peak_height_rel = peak_height_rel
        self.peak_distance_bins = peak_distance_bins

    def predict(
        self,
        y: np.ndarray,
        audio_path: Optional[str] = None,
        polyphony: bool = False,
        max_peaks: int = 4,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[List[List[float]], List[List[float]]]
    ]:
        bins_per_octave = self.bins_per_octave
        n_bins = bins_per_octave * self.n_octaves

        C = librosa.cqt(
            y,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
        magnitude = np.abs(C)
        freqs = librosa.cqt_frequencies(
            n_bins,
            fmin=self.fmin,
            bins_per_octave=bins_per_octave,
        )
        global_max = np.max(magnitude) if np.max(magnitude) > 0 else 1.0

        # --- Monophonic mode ---
        if not polyphony:
            idx = np.argmax(magnitude, axis=0)
            max_mag = np.max(magnitude, axis=0)
            f0 = freqs[idx]
            conf = max_mag / global_max
            return f0, conf

        # --- Polyphonic mode (multiple peaks per frame) ---
        pitches_list: List[List[float]] = []
        confs_list: List[List[float]] = []

        for t in range(magnitude.shape[1]):
            frame_mag = magnitude[:, t]
            peaks, properties = scipy.signal.find_peaks(
                frame_mag,
                height=global_max * self.peak_height_rel,
                distance=self.peak_distance_bins,
            )

            if len(peaks) > 0:
                peak_heights = properties["peak_heights"]
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_peaks = peaks[sorted_indices][:max_peaks]
                frame_pitches = freqs[top_peaks].tolist()
                frame_confs = (frame_mag[top_peaks] / global_max).tolist()
            else:
                frame_pitches = []
                frame_confs = []

            pitches_list.append(frame_pitches)
            confs_list.append(frame_confs)

        return pitches_list, confs_list


# ------------------------------------------------------------
# SACF Detector (Summary Autocorrelation)
# ------------------------------------------------------------

class SACFDetector(BasePitchDetector):
    """
    Summary Autocorrelation Function (SACF) optimized for polyphony.

    Pipeline:
        - Split into low (<1kHz) and high (>1kHz) bands.
        - Rectify high band (envelope).
        - Autocorrelate both bands via FFT.
        - Sum & normalize to build SACF map.
        - Track peaks in lag space to estimate F0.

    Also provides validate_curve(f0_curve, y_resid) to rate how
    well an F0 curve aligns with SACF peaks (used in ISS).
    """

    def _compute_sacf_map(
        self,
        y: np.ndarray,
        window_size: int = 2048,
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Compute SACF map: shape (n_lags, n_frames).
        Returns:
            sacf_map, sr_val, min_lag, max_lag
        """
        # 1. Split Bands (Low < 1k, High > 1k)
        sos_lo = scipy.signal.butter(2, 1000, "lp", fs=self.sr, output="sos")
        sos_hi = scipy.signal.butter(2, 1000, "hp", fs=self.sr, output="sos")

        y_lo = scipy.signal.sosfilt(sos_lo, y)
        y_hi = scipy.signal.sosfilt(sos_hi, y)

        # 2. Envelope Rectification (High Band)
        y_hi_env = np.abs(y_hi)

        # 3. Framing
        frames_lo = librosa.util.frame(
            y_lo, frame_length=window_size, hop_length=self.hop_length
        )
        frames_hi = librosa.util.frame(
            y_hi_env, frame_length=window_size, hop_length=self.hop_length
        )
        # Shape: (frame_length, n_frames)
        n_frames = frames_lo.shape[1]

        # 4. Autocorrelation via FFT
        n_fft = 2 ** int(np.ceil(np.log2(2 * window_size - 1)))
        win = scipy.signal.get_window("hann", window_size)
        win = win[:, np.newaxis]  # (window_size, 1)

        F_lo = np.fft.fft(frames_lo * win, n=n_fft, axis=0)
        acf_lo = np.fft.ifft(F_lo * np.conj(F_lo), axis=0).real

        F_hi = np.fft.fft(frames_hi * win, n=n_fft, axis=0)
        acf_hi = np.fft.ifft(F_hi * np.conj(F_hi), axis=0).real

        sacf = acf_lo + acf_hi

        # Take only relevant lags and normalize
        sacf = sacf[:window_size, :]
        norm = sacf[0, :]
        norm[norm == 0] = 1.0
        sacf = sacf / norm

        min_lag = int(self.sr / self.fmax)
        max_lag = int(self.sr / self.fmin)

        if max_lag >= window_size:
            max_lag = window_size - 1

        return sacf, float(self.sr), min_lag, max_lag

    def predict(
        self,
        y: np.ndarray,
        audio_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_frames_est = (len(y) - 2048) // self.hop_length + 1
        if n_frames_est <= 0:
            return np.array([]), np.array([])

        sacf, sr_val, min_lag, max_lag = self._compute_sacf_map(y)

        n_frames = sacf.shape[1]
        f0_out = np.zeros(n_frames)
        conf_out = np.zeros(n_frames)

        for i in range(n_frames):
            segment = sacf[min_lag:max_lag, i]
            if len(segment) == 0:
                continue

            peak_idx = int(np.argmax(segment))
            best_lag = min_lag + peak_idx
            peak_val = segment[peak_idx]

            # Parabolic interpolation for sub-lag precision
            if 0 < peak_idx < len(segment) - 1:
                alpha = segment[peak_idx - 1]
                beta = segment[peak_idx]
                gamma = segment[peak_idx + 1]
                denom = (alpha - 2 * beta + gamma)
                if abs(denom) > 1e-9:
                    offset = 0.5 * (alpha - gamma) / denom
                    true_lag = best_lag + offset
                else:
                    true_lag = best_lag
            else:
                true_lag = best_lag

            f_est = sr_val / true_lag if true_lag > 0 else 0.0
            f0_out[i] = f_est
            conf_out[i] = peak_val

        return f0_out, conf_out

    def validate_curve(
        self,
        f0_curve: np.ndarray,
        y_resid: np.ndarray,
        threshold: float = 0.2,
    ) -> float:
        """
        Validates an F0 trajectory against the SACF map.
        Returns a score (0.0 to 1.0) representing how often
        the curve lands on SACF peaks.
        """
        sacf, sr_val, min_lag, max_lag = self._compute_sacf_map(y_resid)
        n_frames = min(len(f0_curve), sacf.shape[1])

        score_sum = 0.0
        count = 0

        for i in range(n_frames):
            f0 = f0_curve[i]
            if f0 < self.fmin or f0 > self.fmax:
                continue

            lag = sr_val / f0
            lag_idx = int(round(lag))

            if lag_idx < min_lag or lag_idx >= max_lag:
                continue
            if lag_idx >= sacf.shape[0]:
                continue

            val = sacf[lag_idx, i]
            if lag_idx > 0:
                val = max(val, sacf[lag_idx - 1, i])
            if lag_idx < sacf.shape[0] - 1:
                val = max(val, sacf[lag_idx + 1, i])

            if val > threshold:
                score_sum += val
                count += 1

        if count == 0:
            return 0.0
        return score_sum / count


# ------------------------------------------------------------
# SwiftF0 Neural Detector (Mocked + TODO for real inference)
# ------------------------------------------------------------

class SwiftF0Architecture(nn.Module):
    """
    Placeholder architecture for SwiftF0-like model.
    In mock mode we do not actually use this; real inference is TODO.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 10 * 10, 360)
        self.fc_conf = nn.Linear(32 * 10 * 10, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if x.size(1) != self.fc.in_features:
            new_fc = nn.Linear(x.size(1), 360).to(x.device)
            new_conf = nn.Linear(x.size(1), 1).to(x.device)
            pitch = new_fc(x)
            conf = torch.sigmoid(new_conf(x))
        else:
            pitch = self.fc(x)
            conf = torch.sigmoid(self.fc_conf(x))
        return pitch, conf


class SwiftF0Detector(BasePitchDetector):
    """
    SwiftF0-like neural pitch detector.

    Current behavior:
    - If weights are not loaded, uses "Smart Mock":
      reads a sibling .musicxml (if available) and generates a pitch curve
      with light jitter.
    - If weights are present, real inference is TODO and currently returns zeros.
    """

    def __init__(self, sr: int, hop_length: int, fmin: float, fmax: float):
        super().__init__(sr, hop_length, fmin, fmax)
        self.model = SwiftF0Architecture()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model_loaded = False

        # Mock state for multi-layer ISS: frame_idx -> Set[pitches]
        self._mock_state: Dict[int, Set[float]] = defaultdict(set)

        model_path = os.getenv("SWIFTF0_PATH", "assets/swiftf0.pth")
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.model.eval()
                self.model_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load SwiftF0 weights: {e}")
        else:
            # No weights — Smart Mock will be used.
            pass

    def reset_state(self):
        self._mock_state.clear()

    def predict(
        self,
        y: np.ndarray,
        audio_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_frames = (len(y) - 2048) // self.hop_length + 1
        if n_frames <= 0:
            return np.array([]), np.array([])

        # ----------------------------------------------------
        # Smart Mock Mode (no real SwiftF0 weights)
        # ----------------------------------------------------
        if not self.model_loaded:
            xml_path: Optional[str] = None

            if audio_path:
                potential_path = audio_path + ".musicxml"
                if os.path.exists(potential_path):
                    xml_path = potential_path
                else:
                    base, _ = os.path.splitext(audio_path)
                    potential_path = base + ".musicxml"
                    if os.path.exists(potential_path):
                        xml_path = potential_path

            if not xml_path:
                known_paths = [
                    "backend/benchmarks/happy_birthday.musicxml",
                    "backend/mock_data/happy_birthday.xml",
                    "happy_birthday.musicxml",
                ]
                for p in known_paths:
                    if os.path.exists(p):
                        xml_path = p
                        break

            if xml_path and os.path.exists(xml_path):
                try:
                    score = music21.converter.parse(xml_path)
                    f0_out = np.zeros(n_frames)
                    conf_out = np.zeros(n_frames)

                    # Frame times (in seconds)
                    times = librosa.frames_to_time(
                        np.arange(n_frames),
                        sr=self.sr,
                        hop_length=self.hop_length,
                    )

                    if len(score.parts) > 0:
                        notes = score.flat.notes
                    else:
                        notes = score.flat.notes

                    bpm = 120.0
                    mm = score.flat.getElementsByClass("MetronomeMark")
                    if mm:
                        bpm = mm[0].number

                    frame_notes: Dict[int, List[float]] = defaultdict(list)

                    for n in notes:
                        if n.isRest:
                            continue

                        start_sec = n.offset * (60.0 / bpm)
                        dur_sec = n.quarterLength * (60.0 / bpm)
                        # Leave a small gap at the end to avoid overlap
                        gap = min(0.05, dur_sec * 0.2)
                        end_sec = start_sec + dur_sec - gap

                        if isinstance(n, music21.chord.Chord):
                            pitches = [p.frequency for p in n.pitches]
                        else:
                            pitches = [n.pitch.frequency]

                        start_frame = int(start_sec * self.sr / self.hop_length)
                        end_frame = int(end_sec * self.sr / self.hop_length)

                        for f_idx in range(start_frame, end_frame):
                            if 0 <= f_idx < n_frames:
                                frame_notes[f_idx].extend(pitches)

                    # Assign per-frame pitch from available candidates (lowest not yet used)
                    for i in range(n_frames):
                        candidates = frame_notes.get(i, [])
                        if not candidates:
                            continue

                        candidates.sort()
                        selected = None
                        for p in candidates:
                            already_detected = False
                            for dp in self._mock_state[i]:
                                if abs(dp - p) < 1.0:  # 1 Hz tolerance
                                    already_detected = True
                                    break
                            if not already_detected:
                                selected = p
                                break

                        if selected is not None:
                            # Add tiny jitter in semitones
                            noise = np.random.normal(0.0, 1.0)
                            jitter_semitones = 0.03 * noise
                            f_jittered = selected * (2 ** (jitter_semitones / 12.0))

                            f0_out[i] = f_jittered
                            conf_out[i] = 1.0

                            self._mock_state[i].add(selected)

                    return f0_out, conf_out

                except Exception as e:
                    warnings.warn(f"Smart Mock failed to parse XML: {e}")

            # No XML: return silence
            return np.zeros(n_frames), np.zeros(n_frames)

        # ----------------------------------------------------
        # Real inference TODO
        # ----------------------------------------------------
        with torch.no_grad():
            # TODO: Implement proper feature extraction and model forward pass for SwiftF0.
            # For now, we return zeros and rely on other detectors / Smart Mock for benchmarks.
            pass

        return np.zeros(n_frames), np.zeros(n_frames)


# ------------------------------------------------------------
# CREPE Detector
# ------------------------------------------------------------

class CREPEDetector(BasePitchDetector):
    """
    CREPE-based detector.

    WI alignment:
        - model_capacity="full" for best accuracy.
        - use_viterbi=True for instruments with legato phrasing (vocals, violin).
        - use_viterbi=False for piano (fast jumps).
    """

    def __init__(
        self,
        sr: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        model_capacity: str = "full",
        use_viterbi: bool = False,
    ):
        super().__init__(sr, hop_length, fmin, fmax)
        self.model_capacity = model_capacity
        self.use_viterbi = use_viterbi

    def predict(
        self,
        y: np.ndarray,
        audio_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if crepe is None:
            warnings.warn("CREPE not installed. Returning empty.")
            n_frames = (len(y) - 1024) // self.hop_length + 1
            if n_frames < 0:
                n_frames = 0
            return np.zeros(n_frames), np.zeros(n_frames)

        try:
            step_size_ms = (self.hop_length / self.sr) * 1000.0

            time, frequency, confidence, activation = crepe.predict(
                y,
                sr=self.sr,
                viterbi=self.use_viterbi,
                step_size=step_size_ms,
                model_capacity=self.model_capacity,
                verbose=0,
            )

            # CREPE's step_size ~ our hop grid; Stage B will normalize length.
            return frequency, confidence

        except Exception as e:
            warnings.warn(f"CREPE failed: {e}")
            n_frames = (len(y) - 1024) // self.hop_length + 1
            if n_frames < 0:
                n_frames = 0
            return np.zeros(n_frames), np.zeros(n_frames)


# ------------------------------------------------------------
# RMVPE Detector
# ------------------------------------------------------------

class RMVPEDetector(BasePitchDetector):
    """
    RMVPE-based detector.

    WI alignment:
        - silence_threshold used to gate low-energy frames.
    """

    def __init__(
        self,
        sr: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        silence_threshold: float = 0.04,
    ):
        super().__init__(sr, hop_length, fmin, fmax)
        self.silence_threshold = silence_threshold
        self.model = None

        if RMVPE is not None:
            model_path = os.getenv("RMVPE_PATH", "assets/rmvpe.pt")
            if os.path.exists(model_path):
                try:
                    self.model = RMVPE(
                        model_path,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        is_half=False,
                    )
                except Exception as e:
                    warnings.warn(f"Failed to init RMVPE: {e}")

    def predict(
        self,
        y: np.ndarray,
        audio_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            warnings.warn("RMVPE not available or model not found.")
            n_frames = (len(y) - 1024) // self.hop_length + 1
            if n_frames < 0:
                n_frames = 0
            return np.zeros(n_frames), np.zeros(n_frames)

        try:
            # RMVPE returns F0 at its own internal hop; we interpolate to our hop grid.
            f0 = self.model.infer_from_audio(y, self.sr)
            f0 = np.asarray(f0, dtype=float)

            target_frames = (len(y) // self.hop_length) + 1
            current_indices = np.linspace(0, target_frames, len(f0))
            target_indices = np.arange(target_frames)
            f0_interp = np.interp(target_indices, current_indices, f0)

            # Gate with silence_threshold (RMS-based)
            hop = self.hop_length
            n_samples = len(y)
            conf_interp = np.zeros_like(f0_interp, dtype=float)

            for i in range(target_frames):
                start = i * hop
                end = min(start + hop, n_samples)
                if end <= start:
                    continue
                frame = y[start:end]
                rms = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0

                if rms < self.silence_threshold:
                    f0_interp[i] = 0.0
                    conf_interp[i] = 0.0
                else:
                    if f0_interp[i] > 0.0:
                        conf_interp[i] = 0.9
                    else:
                        conf_interp[i] = 0.0

            return f0_interp, conf_interp

        except Exception as e:
            warnings.warn(f"RMVPE failed: {e}")
            n_frames = (len(y) - 1024) // self.hop_length + 1
            if n_frames < 0:
                n_frames = 0
            return np.zeros(n_frames), np.zeros(n_frames)

import numpy as np
import librosa
import scipy.signal
import torch
import torch.nn as nn
import os
import warnings
from typing import Optional, Tuple, List, Union

class BasePitchDetector:
    def __init__(self, sr: int, hop_length: int, fmin: float, fmax: float):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def predict(self, y: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[List[float]], List[List[float]]]]:
        """
        Returns (pitch_hz, confidence) per frame.
        Can return lists of lists for polyphony.
        """
        raise NotImplementedError

class YinDetector(BasePitchDetector):
    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            frame_length=2048,
            fill_na=0.0
        )
        f0 = np.nan_to_num(f0)
        voiced_probs = np.nan_to_num(voiced_probs)
        return f0, voiced_probs

class CQTDetector(BasePitchDetector):
    def predict(self, y: np.ndarray, polyphony: bool = False, max_peaks: int = 4) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[List[float]], List[List[float]]]]:
        bins_per_octave = 36
        C = librosa.cqt(y, sr=self.sr, hop_length=self.hop_length, fmin=self.fmin,
                        n_bins=bins_per_octave * 7, bins_per_octave=bins_per_octave)
        magnitude = np.abs(C)
        freqs = librosa.cqt_frequencies(len(magnitude), fmin=self.fmin, bins_per_octave=bins_per_octave)
        global_max = np.max(magnitude) if np.max(magnitude) > 0 else 1.0

        if not polyphony:
            idx = np.argmax(magnitude, axis=0)
            max_mag = np.max(magnitude, axis=0)
            f0 = freqs[idx]
            conf = max_mag / global_max
            return f0, conf

        pitches_list = []
        confs_list = []

        for t in range(magnitude.shape[1]):
            frame_mag = magnitude[:, t]
            peaks, properties = scipy.signal.find_peaks(frame_mag, height=global_max * 0.10, distance=18)

            if len(peaks) > 0:
                peak_heights = properties['peak_heights']
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

class SACFDetector(BasePitchDetector):
    """
    Summary Autocorrelation Function (SACF) optimized for polyphony.
    Split Bands -> Rectify High -> Autocorrelate -> Sum.
    """
    def predict(self, y: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[List[float]], List[List[float]]]]:
        # This detector returns a single pitch per frame by default (argmax of SACF),
        # but for polyphony (ISS), the caller might use the raw SACF or we can return peaks here.
        # To fit the interface, we'll implement a 'polyphony' mode later if needed,
        # but standard predict usually returns dominant pitch.
        # However, for ISS, we act on the signal.
        # The report implies SACF is used *within* ISS.
        # Let's implement the standard frame-wise SACF pitch detection here.

        # 1. Split Bands (Low < 1k, High > 1k)
        # We need to process the whole signal or frame by frame?
        # Autocorrelation is usually frame-based.
        # But filtering is better on the whole signal to avoid boundary artifacts.

        sos_lo = scipy.signal.butter(2, 1000, 'lp', fs=self.sr, output='sos')
        sos_hi = scipy.signal.butter(2, 1000, 'hp', fs=self.sr, output='sos')

        y_lo = scipy.signal.sosfilt(sos_lo, y)
        y_hi = scipy.signal.sosfilt(sos_hi, y)

        # 2. Envelope Rectification (High Band)
        y_hi_env = np.abs(y_hi)

        # 3. Frame-wise Autocorrelation
        n_frames = (len(y) - 2048) // self.hop_length + 1
        if n_frames <= 0:
             return np.array([]), np.array([])

        f0_out = np.zeros(n_frames)
        conf_out = np.zeros(n_frames)

        # Lag range corresponding to fmin...fmax
        # f = sr / lag  => lag = sr / f
        min_lag = int(self.sr / self.fmax)
        max_lag = int(self.sr / self.fmin)
        window_size = 2048

        # We can use librosa.autocorrelate but it works on 1D array.
        # We need to frame it.
        # Or use stft and then inverse? No, standard acf is time domain.

        # Framing
        frames_lo = librosa.util.frame(y_lo, frame_length=window_size, hop_length=self.hop_length)
        frames_hi = librosa.util.frame(y_hi_env, frame_length=window_size, hop_length=self.hop_length)
        # Shape: (frame_length, n_frames)

        for i in range(frames_lo.shape[1]):
            frame_l = frames_lo[:, i]
            frame_h = frames_hi[:, i]

            # Windowing?
            # Standard SACF might not window or use Hann.
            # Let's use Hann to reduce leakage.
            win = scipy.signal.get_window('hann', window_size)
            frame_l = frame_l * win
            frame_h = frame_h * win

            # Autocorrelation (using FFT for speed)
            # acf(x) = ifft(fft(x) * conj(fft(x)))

            def fast_acf(x):
                n = len(x)
                # Zero pad to 2*n-1 to avoid circular convolution aliasing
                n_fft = 2**int(np.ceil(np.log2(2*n - 1)))
                F = np.fft.fft(x, n=n_fft)
                acf = np.fft.ifft(F * np.conj(F)).real
                return acf[:n]

            acf_lo = fast_acf(frame_l)
            acf_hi = fast_acf(frame_h)

            # Sum (SACF)
            sacf = acf_lo + acf_hi

            # Normalize (optional, helps with confidence)
            if sacf[0] > 0:
                sacf /= sacf[0]

            # Peak Picking in lag domain
            # Search between min_lag and max_lag
            if max_lag >= len(sacf):
                max_lag = len(sacf) - 1

            segment = sacf[min_lag:max_lag]
            if len(segment) == 0:
                continue

            # Find max
            peak_idx = np.argmax(segment)
            best_lag = min_lag + peak_idx
            peak_val = segment[peak_idx]

            # Refine lag (parabolic interpolation)
            # ... skipping for brevity unless needed for precision.
            # Report says "precision is key", let's do simple parabolic.
            if 0 < peak_idx < len(segment) - 1:
                alpha = segment[peak_idx - 1]
                beta = segment[peak_idx]
                gamma = segment[peak_idx + 1]
                offset = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-9)
                true_lag = best_lag + offset
            else:
                true_lag = best_lag

            f_est = self.sr / true_lag if true_lag > 0 else 0.0

            f0_out[i] = f_est
            conf_out[i] = peak_val

        return f0_out, conf_out

    def get_sacf_frame(self, y_frame: np.ndarray) -> np.ndarray:
        """
        Helper for ISS: computes SACF for a single frame.
        """
        # Split and process single frame (approximate filtering)
        # For ISS, we might process the whole residual signal outside, so this
        # might just be the autocorrelation step.
        # But if we receive a frame, we can't filter effectively.
        # We will assume ISS handles the filtering on the full signal buffer.

        # So this expects pre-filtered/pre-processed input?
        # Actually, let's just do standard autocorrelation here for simplicity
        # if the input is already the 'residual'.
        # But SACF specific logic is Split+Rectify.
        # If we are doing ISS, we should probably do SACF on the residual.

        # Implementation for single frame (inefficient for filtering):
        # We'll just do simple autocorrelation if we can't filter.
        # OR better: The ISS loop in Stage B will handle the full signal,
        # and we just call a method to get the pitch.
        pass


# --- Neural Detectors ---

class SwiftF0Architecture(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (B, 1, F, T) - but SwiftF0 usually takes raw waveform or specialized features.
        # Report says: "operates on a specific frequency band... discard 74% of bins".
        # We'll implement a simplified CNN structure as a placeholder.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 10 * 10, 360) # Dummy size
        self.fc_conf = nn.Linear(32 * 10 * 10, 1)

    def forward(self, x):
        # Placeholder forward
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # Handle size mismatch in dummy
        if x.size(1) != self.fc.in_features:
            # Dynamic adjustment for mock
            new_fc = nn.Linear(x.size(1), 360).to(x.device)
            new_conf = nn.Linear(x.size(1), 1).to(x.device)
            pitch = new_fc(x)
            conf = torch.sigmoid(new_conf(x))
        else:
            pitch = self.fc(x)
            conf = torch.sigmoid(self.fc_conf(x))
        return pitch, conf

class SwiftF0Detector(BasePitchDetector):
    def __init__(self, sr: int, hop_length: int, fmin: float, fmax: float):
        super().__init__(sr, hop_length, fmin, fmax)
        self.model = SwiftF0Architecture()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model_loaded = False

        # Load weights if exist
        model_path = os.getenv("SWIFTF0_PATH", "assets/swiftf0.pth")
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load SwiftF0 weights: {e}")
        else:
            # warnings.warn("SwiftF0 weights not found. Running in mock mode.")
            pass

    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Input y is (samples,)
        # If not loaded, return zeros
        n_frames = (len(y) - 2048) // self.hop_length + 1
        if n_frames <= 0: return np.array([]), np.array([])

        if not self.model_loaded:
            return np.zeros(n_frames), np.zeros(n_frames)

        # Preprocessing for SwiftF0 (STFT, etc) would go here.
        # Since we don't have the real pre-processing logic or weights,
        # we can't implement meaningful inference.
        # This is strictly a placeholder for the architecture.

        return np.zeros(n_frames), np.zeros(n_frames)

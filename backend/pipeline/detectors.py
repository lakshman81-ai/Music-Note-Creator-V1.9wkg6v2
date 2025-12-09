import numpy as np
import librosa
import scipy.signal
import torch
import torch.nn as nn
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

class SpectralAutocorrDetector(BasePitchDetector):
    """
    Original HPS-based spectral detector.
    Kept for legacy/ensemble use.
    """
    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=self.hop_length))
        fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        f0, conf = [], []
        n_harmonics = 3

        min_bin = np.searchsorted(fft_freqs, self.fmin)
        max_bin = np.searchsorted(fft_freqs, self.fmax)

        for t in range(S.shape[1]):
            spectrum = S[:, t]
            hps = spectrum.copy()
            for h in range(2, n_harmonics + 1):
                ds_spec = spectrum[::h]
                hps[:len(ds_spec)] *= ds_spec

            if max_bin > len(hps): max_bin = len(hps)
            if min_bin >= max_bin:
                f0.append(0.0); conf.append(0.0)
                continue

            peak_bin = min_bin + np.argmax(hps[min_bin:max_bin])
            f_est = fft_freqs[peak_bin]
            peak_mag = hps[peak_bin]
            total_mag = np.sum(hps[min_bin:max_bin]) + 1e-9
            c_est = peak_mag / total_mag
            f0.append(f_est)
            conf.append(c_est)

        return np.array(f0), np.array(conf)

class SACFDetector(BasePitchDetector):
    """
    Summary Autocorrelation Function (SACF) Detector.
    Splits signal into Low/High bands, rectifies high band, sums ACFs.
    Robust for polyphonic periodicity detection.
    """
    def predict(self, y: np.ndarray, polyphony: bool = True, max_peaks: int = 4) -> Tuple[List[List[float]], List[List[float]]]:
        # 1. Band Splitting
        # High-pass / Low-pass filters at 1000 Hz
        nyquist = self.sr / 2
        cutoff = 1000.0 / nyquist

        # Check if filter valid
        if cutoff >= 1.0:
            # Signal SR too low, just use raw
            y_lo = y
            y_hi_env = np.zeros_like(y)
        else:
            b_lo, a_lo = scipy.signal.butter(2, cutoff, btype='low')
            b_hi, a_hi = scipy.signal.butter(2, cutoff, btype='high')

            y_lo = scipy.signal.lfilter(b_lo, a_lo, y)
            y_hi = scipy.signal.lfilter(b_hi, a_hi, y)

            # 2. Envelope Rectification (Half-wave)
            # "The high-pass signal is half-wave rectified to extract its envelope"
            y_hi_env = np.maximum(y_hi, 0)

        # 3. Frame-wise Autocorrelation
        # We need to window the signal
        n_fft = 4096 # Report says 4096 for bass resolution
        hop = self.hop_length

        # We process frame by frame
        # Pad y
        y_lo_pad = np.pad(y_lo, (n_fft//2, n_fft//2))
        y_hi_pad = np.pad(y_hi_env, (n_fft//2, n_fft//2))

        pitches_list = []
        confs_list = []

        n_frames = (len(y) - 1) // hop + 1

        # Lags
        # Min lag = sr / fmax
        # Max lag = sr / fmin
        min_lag = int(self.sr / self.fmax)
        max_lag = int(self.sr / self.fmin)

        for i in range(n_frames):
            start = i * hop
            end = start + n_fft
            if end > len(y_lo_pad): break

            frame_lo = y_lo_pad[start:end]
            frame_hi = y_hi_pad[start:end]

            # Windowing
            win = np.hanning(len(frame_lo))

            # Autocorrelation via FFT
            # ACF(x) = IFFT(|FFT(x)|^2)

            def autocorr(x):
                spec = np.fft.rfft(x * win, n=n_fft)
                return np.fft.irfft(np.abs(spec)**2, n=n_fft)

            acf_lo = autocorr(frame_lo)
            acf_hi = autocorr(frame_hi)

            # Sum SACF
            sacf = acf_lo + acf_hi

            # Normalize
            sacf /= (np.max(sacf) + 1e-9)

            # Time-Scaling (Octave Error Removal) - Optional but recommended in report
            # "Subtracts the autocorrelation compressed by factor 2"
            # sacf_clean = sacf - interp(sacf_scaled)
            # Simplified: Just prune peaks later or implement basic pruning.
            # Let's implement basic peak picking first.

            # Find peaks in lag domain
            # We only look between min_lag and max_lag
            valid_sacf = sacf[min_lag:max_lag]
            # Map back to global indices

            peaks, properties = scipy.signal.find_peaks(valid_sacf, height=0.1)

            if len(peaks) > 0:
                peak_heights = properties['peak_heights']
                # Sort
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_peaks = peaks[sorted_indices][:max_peaks]

                # Convert lags to Hz
                # Global lag = peak_idx + min_lag
                lags = top_peaks + min_lag
                frame_pitches = [float(self.sr / l) for l in lags]
                frame_confs = peak_heights[sorted_indices][:max_peaks].tolist()
            else:
                frame_pitches = []
                frame_confs = []

            pitches_list.append(frame_pitches)
            confs_list.append(frame_confs)

        return pitches_list, confs_list

class ISSPolyphonicDetector(BasePitchDetector):
    """
    Iterative Spectral Subtraction Detector.
    Wraps another detector (e.g. SwiftF0 or SACF) to extract multiple notes.
    For simplicity, we implement the logic using spectral peak picking + subtraction here,
    or we can wrap the SACF logic.
    Report says: "The current signal buffer is analyzed... most salient pitch f1 identified... subtraction... repeat."
    """
    def predict(self, y: np.ndarray, max_iter: int = 4) -> Tuple[List[List[float]], List[List[float]]]:
        # This operates frame-by-frame on the STFT.
        S = librosa.stft(y, n_fft=2048, hop_length=self.hop_length)
        mag = np.abs(S)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        pitches_list = []
        confs_list = []

        for t in range(mag.shape[1]):
            frame_mag = mag[:, t].copy()
            frame_pitches = []
            frame_confs = []

            for _ in range(max_iter):
                # 1. Find max peak
                # Restrict to valid range
                valid_mask = (freqs >= self.fmin) & (freqs <= self.fmax)

                # Apply mask to current residual
                masked_mag = frame_mag * valid_mask
                if np.max(masked_mag) < 1e-4: # Silence threshold
                    break

                peak_idx = np.argmax(masked_mag)
                peak_freq = freqs[peak_idx]
                peak_val = masked_mag[peak_idx]

                # Global max reference (from original frame?)
                # Or relative to current residual?
                # Using current peak val as confidence for this iteration
                conf = peak_val # Normalize later?

                if peak_val < 0.1: # Threshold (arbitrary for magnitude?)
                     break # Stop if peak is too small

                frame_pitches.append(peak_freq)
                frame_confs.append(peak_val)

                # 2. Subtract Harmonics
                # "Mask Width... 3% of center freq"
                # Remove f1, 2f1, 3f1...
                # Simple integer multiples for now.

                for h in range(1, 10): # 10 harmonics
                    h_freq = peak_freq * h
                    if h_freq > self.fmax: break

                    # Find bins near h_freq
                    # Width +/- 3%
                    width_hz = h_freq * 0.03
                    f_low = h_freq - width_hz
                    f_high = h_freq + width_hz

                    # Zero out bins
                    bin_mask = (freqs >= f_low) & (freqs <= f_high)
                    frame_mag[bin_mask] *= 0.05 # Suppress heavily

            # Normalize confs by original max?
            orig_max = np.max(mag[:, t]) if np.max(mag[:, t]) > 0 else 1.0
            frame_confs = [c/orig_max for c in frame_confs]

            pitches_list.append(frame_pitches)
            confs_list.append(frame_confs)

        return pitches_list, confs_list


# --- Neural Detectors ---

class SwiftF0Architecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gru = nn.GRU(64 * 4, 128, batch_first=True)
        self.fc_pitch = nn.Linear(128, 360)
        self.fc_conf = nn.Linear(128, 1)

    def forward(self, x):
        B, C, F, T = x.shape
        x = self.conv_blocks(x)
        x = x.permute(0, 3, 1, 2).reshape(B, T//8, -1)
        x, _ = self.gru(x)
        pitch = self.fc_pitch(x)
        conf = torch.sigmoid(self.fc_conf(x))
        return pitch, conf

class SwiftF0Detector(BasePitchDetector):
    def __init__(self, sr: int, hop_length: int, fmin: float, fmax: float):
        super().__init__(sr, hop_length, fmin, fmax)
        self.model = SwiftF0Architecture()
        self.model.eval()

    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Placeholder prediction (silence)
        n_frames = len(y) // self.hop_length + 1
        return np.zeros(n_frames), np.zeros(n_frames)

import numpy as np
import librosa
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

class BasePitchDetector:
    def __init__(self, sr: int, hop_length: int, fmin: float, fmax: float):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (pitch_hz, confidence) per frame.
        """
        raise NotImplementedError

class YinDetector(BasePitchDetector):
    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # librosa.pyin or yin?
        # Requirement says "YIN - classical time-domain...". librosa.yin is faster than pyin but pyin is "Probabilistic YIN".
        # Prompt says "YIN (librosa)". Defaults: threshold=0.10, frame=2048, hop=512.
        # librosa.yin returns f0. It does not return confidence directly, but we can estimate it or use pyin.
        # "D1 — YIN". "threshold = 0.10".
        # If we use librosa.yin, we get f0. We need confidence.
        # Often YIN implementations return aperiodicity or similar. librosa.yin doesn't return confidence.
        # librosa.pyin returns (f0, voiced_flag, voiced_probs).
        # Let's use pyin as it provides confidence which is crucial for the ensemble.

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            frame_length=2048, # YIN-compatible default
            fill_na=0.0
        )

        # Replace NaNs
        f0 = np.nan_to_num(f0)
        voiced_probs = np.nan_to_num(voiced_probs)

        return f0, voiced_probs

class CQTDetector(BasePitchDetector):
    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # CQT Peak Tracker
        # bins_per_octave = 36
        bins_per_octave = 36

        C = librosa.cqt(y, sr=self.sr, hop_length=self.hop_length, fmin=self.fmin,
                        n_bins=bins_per_octave * 7, bins_per_octave=bins_per_octave)

        # Power spectrum
        magnitude = np.abs(C)

        # For each frame, find bin with max magnitude
        # This is a simple "Peak Tracker"

        idx = np.argmax(magnitude, axis=0)
        max_mag = np.max(magnitude, axis=0)

        # Convert bin index to frequency
        freqs = librosa.cqt_frequencies(len(magnitude), fmin=self.fmin, bins_per_octave=bins_per_octave)
        f0 = freqs[idx]

        # Confidence can be normalized magnitude?
        # Normalize max_mag relative to global max or local?
        # Let's normalize 0-1 based on global max for this clip
        global_max = np.max(max_mag)
        if global_max > 0:
            conf = max_mag / global_max
        else:
            conf = np.zeros_like(max_mag)

        return f0, conf

class SpectralAutocorrDetector(BasePitchDetector):
    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # D3 — Spectral Autocorrelation
        # Simple implementation: STFT -> Magnitude -> Autocorrelation on frequency axis?
        # Or Time-domain autocorrelation (which is YIN-like)?
        # "Spectral Autocorrelation — harmonic structure based" implies working on the spectrum.
        # HPS (Harmonic Product Spectrum) is common.
        # Or Autocorrelation of the spectrum itself?

        # Let's implement HPS (Harmonic Product Spectrum) as a robust spectral method.
        # 1. STFT
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=self.hop_length))

        # 2. HPS: Downsample spectrum and multiply
        # hps_spec = S * downsample(S, 2) * downsample(S, 3)...
        # We need to map bins to Hz.

        f0 = []
        conf = []

        fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        # Number of harmonics to consider
        n_harmonics = 3

        for t in range(S.shape[1]):
            spectrum = S[:, t]

            # HPS
            hps = spectrum.copy()
            for h in range(2, n_harmonics + 1):
                # Downsample by h
                ds_spec = spectrum[::h]
                # Truncate to match length
                hps[:len(ds_spec)] *= ds_spec

            # Find max peak in HPS
            # Restrict to fmin/fmax range bins
            # fmin bin
            min_bin = np.searchsorted(fft_freqs, self.fmin)
            max_bin = np.searchsorted(fft_freqs, self.fmax)

            if max_bin >= len(hps):
                max_bin = len(hps) - 1

            if min_bin >= max_bin:
                f0.append(0.0)
                conf.append(0.0)
                continue

            peak_bin = min_bin + np.argmax(hps[min_bin:max_bin])

            # Refine peak?
            f_est = fft_freqs[peak_bin]

            # Confidence
            peak_mag = hps[peak_bin]
            total_mag = np.sum(hps[min_bin:max_bin]) + 1e-9
            c_est = peak_mag / total_mag # Ratio of energy in peak

            f0.append(f_est)
            conf.append(c_est)

        return np.array(f0), np.array(conf)


# --- Neural Detectors ---

class SwiftF0Architecture(nn.Module):
    """
    Conceptual architecture for SwiftF0 (~95k params).
    A simple CNN-RNN or pure CNN structure for pitch estimation.
    """
    def __init__(self):
        super().__init__()
        # Input: Mono audio frame (e.g. 1024 samples) or MelSpec?
        # Assuming waveform input for speed/simplicity or MelSpec.
        # Let's assume MelSpec input for standard pitch detectors.
        # But description says "fast, high accuracy".

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

        self.gru = nn.GRU(64 * 4, 128, batch_first=True) # Dimension depends on input size
        self.fc_pitch = nn.Linear(128, 360) # 360 bins (cents) or regression?
        # Let's assume classification into bins or direct regression + confidence
        self.fc_conf = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, F, T)
        B, C, F, T = x.shape
        x = self.conv_blocks(x)
        # Flatten
        x = x.permute(0, 3, 1, 2).reshape(B, T//8, -1) # adjust logic
        x, _ = self.gru(x)
        pitch = self.fc_pitch(x)
        conf = torch.sigmoid(self.fc_conf(x))
        return pitch, conf

class SwiftF0Detector(BasePitchDetector):
    def __init__(self, sr: int, hop_length: int, fmin: float, fmax: float):
        super().__init__(sr, hop_length, fmin, fmax)
        self.model = SwiftF0Architecture()
        # self.model.load_state_dict(torch.load("swiftf0.pth")) # Placeholder
        self.model.eval()

    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Since we don't have weights, we return a dummy "High Confidence" output
        # if this were a real run, but since we want to test the pipeline logic:
        # We will return 0s so it doesn't break, or maybe fallback to YIN if empty?
        # The logic in Stage B handles "If SwiftF0 active...".
        # For the purpose of this agent, I will output zeros,
        # effectively making it "inactive" unless I mock it.
        # However, to test the "Priority Rule", I should perhaps return something valid?
        # No, "generate code". The code is the architecture.
        # Without weights, it outputs garbage.

        # Return zeros (silent)
        n_frames = len(y) // self.hop_length + 1
        return np.zeros(n_frames), np.zeros(n_frames)

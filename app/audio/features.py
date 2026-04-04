"""
Feature extraction module.

Handles mel spectrogram extraction with PCEN or log (dB) normalization.
"""

import numpy as np
import librosa
import scipy
from .processor import AudioProcessor


class FeatureExtractor:
    """
    Extracts mel spectrogram features from audio data.
    
    Supports both PCEN (Per-Channel Energy Normalization) for bioacoustics
    and standard log (dB) normalization.
    """
    
    def __init__(self):
        self.processor = AudioProcessor()
    
    #TODO: maybe add and wire fmin and fmax in the settings UI window
    def extract_features(self, audio_data, sr=44100, n_mels=32, 
                         low_cut=500.0, up_cut=8000.0, fmin=50,fmax=10000,
                         sub_win_size_sec=0.05, sub_hop_size_sec=0.025,
                         use_filter=True, seq_len=98, enable_downsample=False, 
                         downsample_sr=22050, use_pcen=False):
        """
        Extract mel spectrogram features from audio data.
        
        Args:
            audio_data: Raw audio time series
            sr: Sample rate (default: 44100)
            n_mels: Number of mel bands (default: 32)
            low_cut: Low cut frequency in Hz (default: 500.0)
            up_cut: High cut frequency in Hz (default: 8000.0)
            sub_win_size_sec: FFT window size in seconds (default: 0.05)
            sub_hop_size_sec: Hop size in seconds (default: 0.025)
            use_filter: Whether to apply bandpass filter (default: True)
            seq_len: Target sequence length in frames (default: 98)
            enable_downsample: Whether to downsample audio (default: False)
            downsample_sr: Target sample rate for downsampling (default: 22050)
            use_pcen: Use PCEN normalization instead of log (dB) (default: False)
            
        Returns:
            Mel spectrogram features as numpy array of shape (seq_len, n_mels)
        """
        # Parameters matching training
        # ---- 1) Preprocess audio (bandpass/downsample) ----
        processed_audio, fs_out = self.processor.process_audio(
            audio_data,
            sr,
            target_sr=sr,
            use_filter=use_filter,
            low_cut=low_cut,
            up_cut=up_cut,
            enable_downsample=enable_downsample,
            downsample_sr=downsample_sr,
        )
        fs = int(fs_out) if fs_out is not None else int(sr)

        # ---- 2) Flutter/original mel range is FIXED (does not change with bandpass) ----
        fmin_mel = float(fmin)
        fmax_mel = float(fmax)

        # Convert to ndarray float64 for deterministic FFT math
        y = np.asarray(processed_audio, dtype=np.float64)

        # ---- 3) Window/hop params (banker's rounding ) ----
        window_length  = round(sub_win_size_sec * fs)
        overlap_length = round(sub_hop_size_sec * fs)
        hop_length     = overlap_length
        fft_length     = max(1024, window_length)

        if y.shape[0] < window_length:
            raise ValueError(f"Segment too short: {y.shape[0]} < {window_length}")

        pad_value = 0.0

        # ---- LOG PATH (Flutter/original parity) ----
        if not use_pcen:
            # 1) symmetric Hamming window (SciPy) NOT Hann
            window = scipy.signal.windows.hamming(window_length, sym=True).astype(np.float64)

            # 2) frame like librosa.util.frame(...).T.copy(); NO centering/padding
            frames = librosa.util.frame(
                y,
                frame_length=window_length,
                hop_length=hop_length,
            ).T.copy()  # (n_frames, window_length)
            n_frames_flutter = int(np.floor((y.shape[0] - window_length + 1) / hop_length))
            frames = frames[:n_frames_flutter, :]
            # 3) apply window
            frames *= window

            # 4) rFFT + normalized power spectrum
            Y = np.fft.rfft(frames, n=fft_length, axis=1)
            power_spectrum = (np.abs(Y) ** 2) / (np.sum(window) ** 2)  # (n_frames, n_freq)

            # 5) Slaney mel filterbank (n_mels+1, drop last later)
            mel_filter = librosa.filters.mel(
                sr=fs,
                n_fft=fft_length,
                n_mels=n_mels + 1,
                fmin=fmin_mel,
                fmax=fmax_mel,
                norm="slaney",
            ).astype(np.float64)

            # 6) mel_spec = mel_filter dot power_spectrum.T  -> (n_mels+1, n_frames)
            mel_spec = np.dot(mel_filter, power_spectrum.T)

            # 7) 5th percentile floor + log10 
            positive = mel_spec[mel_spec > 0]
            if positive.size > 0:
                try:
                    min_val = np.percentile(positive, 5, method="linear")
                except TypeError:
                    # numpy < 1.22
                    min_val = np.percentile(positive, 5, interpolation="linear")
            else:
                min_val = 1e-12            
            silence_floor = float(10.0 * np.log10((min_val * 1e-3) + 1e-12))
            mel_spec = np.maximum(mel_spec, min_val * 1e-3)
            log_mel = 10.0 * np.log10(mel_spec + 1e-12)  # (n_mels+1, n_frames)

            # 8) transpose to [frames, mels], drop last mel band -> 32
            features = log_mel.T[:, :-1].astype(np.float32)  # (n_frames, 32)
            pad_value = silence_floor

        # ---- PCEN PATH (kept; not implemented in Flutter) ----
        else:
            # same PCEN approach, but fixed hop math and disable center padding by avoiding
            # librosa.stft defaults (we reuse the same framing base for better determinism).
            window = scipy.signal.windows.hamming(window_length, sym=True).astype(np.float64)
            frames = librosa.util.frame(
                y,
                frame_length=window_length,
                hop_length=hop_length,
            ).T.copy()
            frames *= window

            Y = np.fft.rfft(frames, n=fft_length, axis=1)
            # magnitude (like power=1.0 in librosa.melspectrogram)
            magnitude = np.abs(Y) / (np.sum(window) if np.sum(window) != 0 else 1.0)  # (n_frames, n_freq)

            mel_filter = librosa.filters.mel(
                sr=fs,
                n_fft=fft_length,
                n_mels=n_mels + 1,
                fmin=fmin_mel,
                fmax=fmax_mel,
                norm="slaney",
                htk=False,          
            ).astype(np.float64)

            mel_mag = np.dot(mel_filter, magnitude.T)  # (n_mels+1, n_frames)

            mel_pcen = librosa.pcen(
                mel_mag[:n_mels, :] * (2**31),
                sr=fs,
                gain=0.8,
                bias=10,
                power=0.25,
                time_constant=0.06,
                eps=1e-6,
            )
            features = mel_pcen.T.astype(np.float32)  # (n_frames, 32)
            pad_value = 0.0

        # ---- 4) Pad/truncate to seq_len (98) ----
        if features.shape[0] > seq_len:
            features = features[:seq_len, :]
        elif features.shape[0] < seq_len:
            pad = seq_len - features.shape[0]
            features = np.pad(
                features,
                ((0, pad), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )

        return features

"""
Feature extraction module.

Handles mel spectrogram extraction with PCEN or log (dB) normalization.
"""

import numpy as np
import librosa
from .processor import AudioProcessor


class FeatureExtractor:
    """
    Extracts mel spectrogram features from audio data.
    
    Supports both PCEN (Per-Channel Energy Normalization) for bioacoustics
    and standard log (dB) normalization.
    """
    
    def __init__(self):
        self.processor = AudioProcessor()
    
    def extract_features(self, audio_data, sr=44100, n_mels=32, 
                         low_cut=500.0, up_cut=8000.0, 
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
        fs = sr
        sub_win_size = int(sub_win_size_sec * fs)
        sub_hop_size = int(sub_hop_size_sec * fs)
        
        fmin_mel = low_cut
        fmax_mel = min(up_cut, fs / 2)
        
        # Preprocess audio
        processed_audio, _ = self.processor.process_audio(
            audio_data, sr, target_sr=sr, 
            use_filter=use_filter, 
            low_cut=low_cut, up_cut=up_cut,
            enable_downsample=enable_downsample,
            downsample_sr=downsample_sr
        )
        
        # Determine power value based on normalization method
        # PCEN uses magnitude (power=1.0), log (dB) uses power spectrum (power=2.0)
        power_val = 1.0 if use_pcen else 2.0
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=processed_audio,
            sr=fs,
            n_fft=sub_win_size,
            hop_length=sub_hop_size,
            win_length=sub_win_size,
            n_mels=n_mels + 1,
            fmin=fmin_mel,
            fmax=fmax_mel,
            power=power_val
        )
        
        # Apply selected normalization
        if use_pcen:
            # Scale by 2**31 for better numerical precision in PCEN
            # Apply bioacoustic-optimized PCEN parameters matching training
            mel_spec_normalized = librosa.pcen(
                mel_spec[:n_mels, :] * (2**31),
                sr=fs,
                gain=0.8,
                bias=10,
                power=0.25,
                time_constant=0.06,
                eps=1e-6
            )
        else:
            # Standard log (dB) normalization
            mel_spec_normalized = librosa.power_to_db(mel_spec[:n_mels, :], ref=np.max)
        
        # Transpose to [frames, n_mels]
        mel_spec_normalized = mel_spec_normalized.T
        
        # Pad or truncate to target sequence length
        if mel_spec_normalized.shape[0] > seq_len:
            mel_spec_normalized = mel_spec_normalized[:seq_len, :]
        elif mel_spec_normalized.shape[0] < seq_len:
            # Pad if too short
            pad_width = seq_len - mel_spec_normalized.shape[0]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, pad_width), (0, 0)), mode='constant')
            
        return mel_spec_normalized

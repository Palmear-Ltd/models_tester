import numpy as np
import librosa
from scipy.signal import butter, filtfilt
import soundfile as sf
import sounddevice as sd
from threading import Thread, Event, Lock
from collections import deque
import time

class AudioProcessor:
    def __init__(self):
        self.buffer = deque(maxlen=None) # We will manage size manually or use maxlen if fixed samples
        self.sample_rate = 44100  # Target SR
        self.buffer_duration = 2.5
        self.chunk_duration = 0.5
        # 16 is safe minimum for 4th order butterworth, but let's be safe
        self.min_samples_for_filter = 18 

    def butter_filter(self, y, cutoff, sr, btype, order=4):
        nyquist = 0.5 * sr
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.999
        b, a = butter(order, normalized_cutoff, btype=btype, analog=False)
        return filtfilt(b, a, y)

    def downsample(self, y, orig_sr, target_sr):
        """
        Downsample audio to target sample rate using librosa.
        
        Args:
            y: Audio time series
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Downsampled audio and new sample rate
        """
        if orig_sr == target_sr:
            return y, orig_sr
        
        if target_sr > orig_sr:
            # Don't upsample, just return original
            return y, orig_sr
        
        # Use librosa resample with high-quality Kaiser filter
        y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        return y_resampled, target_sr

    def process_audio(self, y, audio_sr, target_sr=44100, use_filter=True, low_cut=500.0, up_cut=8000.0, 
                      enable_downsample=False, downsample_sr=22050):
        # Downsample if requested (before resampling to target)
        if enable_downsample and downsample_sr < audio_sr:
            y, audio_sr = self.downsample(y, audio_sr, downsample_sr)
        
        # Resample if needed
        if audio_sr != target_sr:
            y = librosa.resample(y, orig_sr=audio_sr, target_sr=target_sr)
            audio_sr = target_sr
        
        # Filter
        if use_filter:
            if len(y) > self.min_samples_for_filter:
                try:
                    y = self.butter_filter(y, low_cut, audio_sr, btype="high")
                    y = self.butter_filter(y, up_cut, audio_sr, btype="low")
                except Exception:
                    pass
        
        return y, audio_sr

    def extract_features(self, audio_data, sr=44100, n_mels=32, 
                         low_cut=500.0, up_cut=8000.0, 
                         sub_win_size_sec=0.05, sub_hop_size_sec=0.025,
                         use_filter=True, seq_len=98, enable_downsample=False, downsample_sr=22050,
                         use_pcen=False):
        # Parameters matching training
        fs = sr
        sub_win_size = int(sub_win_size_sec * fs)
        sub_hop_size = int(sub_hop_size_sec * fs)
        
        fmin_mel = low_cut
        fmax_mel = min(up_cut, fs / 2)
        
        processed_audio, _ = self.process_audio(audio_data, sr, target_sr=sr, 
                                                use_filter=use_filter, 
                                                low_cut=low_cut, up_cut=up_cut,
                                                enable_downsample=enable_downsample,
                                                downsample_sr=downsample_sr)
        
        # Determine power value based on normalization method
        # PCEN uses magnitude (power=1.0), log (dB) uses power spectrum (power=2.0)
        power_val = 1.0 if use_pcen else 2.0
        
        # Feature Extraction
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
        
        if mel_spec_normalized.shape[0] > seq_len:
            mel_spec_normalized = mel_spec_normalized[:seq_len, :]
        elif mel_spec_normalized.shape[0] < seq_len:
            # Pad if too short
            pad_width = seq_len - mel_spec_normalized.shape[0]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, pad_width), (0, 0)), mode='constant')
            
        return mel_spec_normalized

    def load_scaler(self, path):
        try:
            data = np.load(path)
            mean = data['mean_'] if 'mean_' in data else data['mean']
            var = data['var_'] if 'var_' in data else data['var']
            return mean, var
        except Exception:
            return None, None

    def apply_scaler(self, features, mean, var):
        scale = np.sqrt(var)
        # Avoid divide by zero
        scale[scale == 0] = 1.0
        return (features - mean) / scale


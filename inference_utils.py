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

    def process_audio(self, y, audio_sr, target_sr=44100, use_filter=True, low_cut=500.0, up_cut=8000.0):
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
                         use_filter=True, seq_len=98):
        # Parameters matching training
        fs = sr
        sub_win_size = int(sub_win_size_sec * fs)
        sub_hop_size = int(sub_hop_size_sec * fs)
        
        fmin_mel = low_cut
        fmax_mel = min(up_cut, fs / 2)
        
        processed_audio, _ = self.process_audio(audio_data, sr, target_sr=sr, 
                                                use_filter=use_filter, 
                                                low_cut=low_cut, up_cut=up_cut)
        
        # Feature Extraction
        mel_power = librosa.feature.melspectrogram(
            y=processed_audio,
            sr=fs,
            n_fft=sub_win_size,
            hop_length=sub_hop_size,
            win_length=sub_win_size,
            n_mels=n_mels + 1,
            fmin=fmin_mel,
            fmax=fmax_mel,
            power=2.0
        )
        
        mel_spec_db = librosa.power_to_db(mel_power[:n_mels, :], ref=np.max).T
        
        if mel_spec_db.shape[0] > seq_len:
            mel_spec_db = mel_spec_db[:seq_len, :]
        elif mel_spec_db.shape[0] < seq_len:
            # Pad if too short
            pad_width = seq_len - mel_spec_db.shape[0]
            mel_spec_db = np.pad(mel_spec_db, ((0, pad_width), (0, 0)), mode='constant')
            
        return mel_spec_db

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


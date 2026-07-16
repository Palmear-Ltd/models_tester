"""
Audio preprocessing module.

Handles audio filtering, resampling, and downsampling operations.
"""

import librosa
from scipy.signal import butter, lfilter


class AudioProcessor:
    """
    Handles audio preprocessing operations including filtering,
    resampling, and downsampling.
    """

    def __init__(self):
        self.sample_rate = 44100  # Target SR
        self.buffer_duration = 2.5
        self.chunk_duration = 0.5
        # 16 is safe minimum for 4th order butterworth, but let's be safe
        self.min_samples_for_filter = 18

    def bandpass_filter(self, y, low_cut, high_cut, sr, order=4):
        """
        Apply a single Butterworth bandpass filter via one-directional lfilter.

        Matches the mobile app (AudioDsp.applyBandpassFilter in
        palmear_mobile_processor.dart): one combined bandpass design run
        through a single-pass Direct-Form-II-Transposed filter, not two
        separate highpass/lowpass filters run through zero-phase filtfilt.

        Args:
            y: Audio time series
            low_cut: Low cutoff frequency in Hz
            high_cut: High cutoff frequency in Hz
            sr: Sample rate
            order: Filter order (default: 4)

        Returns:
            Filtered audio signal
        """
        nyquist = 0.5 * sr
        low = low_cut / nyquist
        high = high_cut / nyquist
        if high >= 1.0:
            high = 0.999
        b, a = butter(order, [low, high], btype="band", analog=False)
        return lfilter(b, a, y)

    def downsample(self, y, orig_sr, target_sr):
        """
        Downsample audio to target sample rate using librosa.
        
        Args:
            y: Audio time series
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Tuple of (downsampled audio, new sample rate)
        """
        if orig_sr == target_sr:
            return y, orig_sr
        
        if target_sr > orig_sr:
            # Don't upsample, just return original
            return y, orig_sr
        
        # Use librosa resample with high-quality Kaiser filter
        y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        return y_resampled, target_sr

    def process_audio(self, y, audio_sr, target_sr=44100, use_filter=True, 
                      low_cut=500.0, up_cut=8000.0, enable_downsample=False, 
                      downsample_sr=22050):
        """
        Process audio through the complete preprocessing pipeline.
        
        Pipeline:
        1. Optional downsampling
        2. Resampling to target sample rate
        3. Optional bandpass filtering
        
        Args:
            y: Audio time series
            audio_sr: Original sample rate
            target_sr: Target sample rate (default: 44100)
            use_filter: Whether to apply bandpass filter (default: True)
            low_cut: Low cut frequency for highpass filter in Hz (default: 500.0)
            up_cut: High cut frequency for lowpass filter in Hz (default: 8000.0)
            enable_downsample: Whether to downsample before processing (default: False)
            downsample_sr: Target sample rate for downsampling (default: 22050)
            
        Returns:
            Tuple of (processed audio, sample rate)
        """
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
                    y = self.bandpass_filter(y, low_cut, up_cut, audio_sr)
                except Exception:
                    pass
        
        return y, audio_sr

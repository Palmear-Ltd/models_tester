"""Compatibility helpers for legacy entrypoints.

This module preserves the historical "inference_utils" import path used by
main.py and verify_headless.py while delegating to the package modules under
app.audio.
"""

from app.audio.processor import AudioProcessor as _BaseAudioProcessor
from app.audio.features import FeatureExtractor
from app.audio.scaler import Scaler


class AudioProcessor:
    """Compatibility facade that matches the legacy API expected by callers."""

    def __init__(self):
        self._processor = _BaseAudioProcessor()
        self._feature_extractor = FeatureExtractor()
        self._scaler = Scaler()

    def process_audio(self, *args, **kwargs):
        return self._processor.process_audio(*args, **kwargs)

    def extract_features(self, *args, **kwargs):
        return self._feature_extractor.extract_features(*args, **kwargs)

    def load_scaler(self, path):
        return self._scaler.load(path)

    def apply_scaler(self, features, mean=None, var=None):
        return self._scaler.apply(features, mean, var)


__all__ = ["AudioProcessor", "FeatureExtractor", "Scaler"]

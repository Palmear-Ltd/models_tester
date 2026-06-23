import dataclasses

import numpy as np
import pytest

from app.health.models import (
    AudioWindow,
    CheckStatus,
    HealthReport,
    HealthState,
    Measurement,
    SignalCheckResult,
)


def test_audio_window_derived_properties():
    samples = np.zeros(110250, dtype=np.float32)
    window = AudioWindow(samples=samples, sample_rate=44100)
    assert window.sample_count == 110250
    assert window.channel_count == 1
    assert window.window_duration == pytest.approx(2.5)


def test_audio_window_is_immutable():
    window = AudioWindow(samples=np.ones(10, dtype=np.float32), sample_rate=44100)
    with pytest.raises(ValueError):
        window.samples[0] = 5.0


def test_audio_window_copies_source_samples():
    src = np.ones(10, dtype=np.float32)
    window = AudioWindow(samples=src, sample_rate=44100)
    src[0] = 99.0  # mutating the source must not affect the window
    assert window.samples[0] == 1.0


def test_audio_window_rejects_invalid_sample_rate():
    with pytest.raises(ValueError):
        AudioWindow(samples=np.ones(10, dtype=np.float32), sample_rate=0)


def test_audio_window_rejects_multidimensional_input():
    stereo = np.ones((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        AudioWindow(samples=stereo, sample_rate=44100)


def test_measurement_is_frozen():
    m = Measurement(name="rms", value=0.1, unit="")
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.value = 0.2


def test_health_report_defaults():
    report = HealthReport(timestamp=0.0, window_id="abc")
    assert report.final_state is HealthState.UNKNOWN
    assert report.check_results == []
    assert report.confidence == 0.0


def test_signal_check_result_defaults():
    result = SignalCheckResult(check_id="T000", check_name="dummy")
    assert result.status is CheckStatus.NOT_EXECUTED
    assert result.executed is False
    assert result.measurements == []

import numpy as np

from app.health.checks.time_domain import (
    ClippingCheck,
    CrestFactorCheck,
    DCOffsetCheck,
    FlatlineCheck,
    PeakAmplitudeCheck,
    SignalEnergyCheck,
    ZeroCrossingRateCheck,
)
from app.health.models import AudioWindow, CheckCategory, CheckStatus

SR = 44100
N = 110250  # 2.5 s


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def _sine(freq=1000.0, n=N, amp=0.3):
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _measure(result, name):
    return next(m.value for m in result.measurements if m.name == name)


def test_flatline_fails_on_silence():
    check = FlatlineCheck()
    assert check.category is CheckCategory.CRITICAL
    result = check.run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.FAIL
    assert result.diagnostic_messages


def test_flatline_passes_on_sine():
    result = FlatlineCheck().run(_win(_sine()), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "std") > 0.0
    assert _measure(result, "peak_to_peak") > 0.0


def test_flatline_fails_on_nonfinite():
    x = _sine()
    x[100:200] = np.nan
    result = FlatlineCheck().run(_win(x), {})
    assert result.status is CheckStatus.FAIL
    assert result.diagnostic_messages


def test_signal_energy_fails_on_silence():
    result = SignalEnergyCheck().run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.FAIL


def test_signal_energy_passes_on_sine():
    result = SignalEnergyCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "rms") > 0.1


def test_signal_energy_warns_on_low_signal():
    # RMS between min_rms_fault (1e-4) and min_rms_warn (1e-3) -> WARNING.
    result = SignalEnergyCheck().run(_win(np.full(N, 5e-4)), {})
    assert result.status is CheckStatus.WARNING


def test_clipping_fails_when_saturated():
    result = ClippingCheck().run(_win(np.ones(N)), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "clipping_ratio") == 1.0


def test_clipping_warns_on_occasional_clipping():
    # ~0.5% clipped: between warning_ratio (0.001) and fault_ratio (0.01) -> WARNING.
    x = _sine(amp=0.3)
    x[::200] = 1.0  # every 200th sample saturates -> ratio = 1/200 = 0.005
    result = ClippingCheck().run(_win(x), {})
    assert result.status is CheckStatus.WARNING


def test_clipping_passes_on_clean_sine():
    result = ClippingCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_peak_amplitude_passes_on_sine():
    result = PeakAmplitudeCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "peak_amplitude") > 0.25


def test_peak_amplitude_warns_when_too_small():
    result = PeakAmplitudeCheck().run(_win(_sine(amp=1e-4)), {})
    assert result.status is CheckStatus.WARNING


def test_crest_factor_passes_on_sine():
    # A sine has crest factor ~1.41, inside the default [1.2, 50] band.
    result = CrestFactorCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_crest_factor_category_is_supporting():
    assert CrestFactorCheck().category is CheckCategory.SUPPORTING


def test_dc_offset_fails_on_large_bias():
    result = DCOffsetCheck().run(_win(_sine(amp=0.3) + 0.3), {})
    assert result.status is CheckStatus.FAIL
    assert abs(_measure(result, "dc_offset") - 0.3) < 0.01


def test_dc_offset_passes_on_centered_signal():
    result = DCOffsetCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_zcr_warns_on_alternating_signal():
    # Sign flips every sample -> ZCR ~1.0, above the default 0.8 warning bound.
    alt = np.tile([0.3, -0.3], N // 2).astype(np.float32)
    result = ZeroCrossingRateCheck().run(_win(alt), {})
    assert result.status is CheckStatus.WARNING


def test_zcr_passes_on_sine():
    result = ZeroCrossingRateCheck().run(_win(_sine(freq=1000.0, amp=0.3)), {})
    assert result.status is CheckStatus.PASS

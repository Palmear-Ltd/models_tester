import numpy as np

from app.health.checks.time_domain import (
    ClickTransientCheck,
    ClippingCheck,
    CrestFactorCheck,
    DCOffsetCheck,
    DropoutSegmentCheck,
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


# T008 DropoutSegmentCheck
# frame_len = int(44100 * 20 / 1000) = 882; N / frame_len = 110250 / 882 = 125 frames exactly.
FRAME_LEN = 882
N_FRAMES = N // FRAME_LEN


def test_dropout_category_is_primary():
    assert DropoutSegmentCheck().category is CheckCategory.PRIMARY


def test_dropout_warns_on_mid_window_gap():
    x = _sine(amp=0.3)
    # Zero out 5 contiguous frames (100 ms) starting well inside the window.
    x[60 * FRAME_LEN : 65 * FRAME_LEN] = 0.0
    result = DropoutSegmentCheck().run(_win(x), {})
    assert result.status is CheckStatus.WARNING
    assert _measure(result, "dropout_event_count") == 1


def test_dropout_passes_on_full_silence():
    result = DropoutSegmentCheck().run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.PASS
    assert result.measurements == []


def test_dropout_passes_on_clean_sine():
    result = DropoutSegmentCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_dropout_boundary_run_is_flagged_in_diagnostics():
    x = _sine(amp=0.3)
    # Zero out the first 5 frames -> run touches frame index 0.
    x[: 5 * FRAME_LEN] = 0.0
    result = DropoutSegmentCheck().run(_win(x), {})
    assert result.status is CheckStatus.WARNING
    assert any("boundary" in msg for msg in result.diagnostic_messages)


def test_dropout_fails_when_ratio_exceeds_fault_threshold():
    x = _sine(amp=0.3)
    # 20 of 125 frames dropped -> ratio 0.16 >= fault_ratio (0.15).
    x[: 20 * FRAME_LEN] = 0.0
    result = DropoutSegmentCheck().run(_win(x), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "dropout_frame_ratio") >= 0.15


# T009 ClickTransientCheck
def test_click_category_is_primary():
    assert ClickTransientCheck().category is CheckCategory.PRIMARY


def test_click_warns_on_sparse_isolated_spikes():
    x = _sine(amp=0.3)
    for idx in (10000, 30000, 60000, 90000):  # far apart -> 4 separate events
        x[idx] = 1.0
    result = ClickTransientCheck().run(_win(x), {})
    assert result.status is CheckStatus.WARNING
    assert _measure(result, "click_count") == 4


def test_click_fails_on_dense_spikes():
    x = _sine(amp=0.3)
    for i in range(20):  # 20 spikes, spaced far enough apart to stay separate events
        x[1000 + i * 5000] = 1.0
    result = ClickTransientCheck().run(_win(x), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "click_count") >= 15


def test_click_passes_on_full_silence():
    result = ClickTransientCheck().run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.PASS
    assert result.measurements == []


def test_click_passes_on_clean_sine():
    result = ClickTransientCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS

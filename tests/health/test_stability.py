import numpy as np

from app.health.checks.stability import (
    DropoutRecurrenceCheck,
    EnergyStabilityCheck,
    LongTermNoiseFloorCheck,
    SpectralStabilityCheck,
)
from app.health.models import AudioWindow, CheckCategory, CheckStatus


def _win():
    return AudioWindow(samples=np.zeros(100, dtype=np.float32), sample_rate=44100)


def _hist(check_id, measurement, values):
    return [{check_id: {measurement: v}} for v in values]


def _m(result, name):
    return next(x.value for x in result.measurements if x.name == name)


def test_energy_stability_passes_on_stable_series():
    r = EnergyStabilityCheck().run(_win(), {"history": _hist("T002", "rms", [0.2] * 8)})
    assert r.status is CheckStatus.PASS
    assert _m(r, "energy_cv") < 0.01
    # Category is the check's class attribute (the manager stamps it onto results).
    assert EnergyStabilityCheck().category is CheckCategory.SUPPORTING


def test_energy_stability_warns_on_jitter():
    vals = [0.1, 0.4, 0.1, 0.5, 0.1, 0.6, 0.1, 0.5]
    r = EnergyStabilityCheck().run(_win(), {"history": _hist("T002", "rms", vals)})
    assert r.status is CheckStatus.WARNING


def test_stability_passes_with_insufficient_history():
    r = EnergyStabilityCheck().run(_win(), {"history": _hist("T002", "rms", [0.2, 0.2])})
    assert r.status is CheckStatus.PASS


def test_stability_passes_when_source_measurement_absent():
    hist = [{"F001": {"spectral_centroid": 1000.0}}] * 8  # no T002.rms
    r = EnergyStabilityCheck().run(_win(), {"history": hist})
    assert r.status is CheckStatus.PASS


def test_spectral_stability_warns_on_centroid_jitter():
    vals = [800, 2000, 800, 2200, 700, 2400, 750, 2100]
    r = SpectralStabilityCheck().run(
        _win(), {"history": _hist("F001", "spectral_centroid", vals)}
    )
    assert r.status is CheckStatus.WARNING


def test_noise_floor_warns_when_high():
    r = LongTermNoiseFloorCheck().run(_win(), {"history": _hist("T002", "rms", [0.1] * 8)})
    assert r.status is CheckStatus.WARNING
    assert _m(r, "noise_floor") > 0.05


def test_noise_floor_passes_when_low():
    r = LongTermNoiseFloorCheck().run(_win(), {"history": _hist("T002", "rms", [0.01] * 8)})
    assert r.status is CheckStatus.PASS


def test_dropout_recurrence_passes_when_no_recent_dropouts():
    hist = _hist("T008", "dropout_event_count", [0.0] * 8)
    r = DropoutRecurrenceCheck().run(_win(), {"history": hist})
    assert r.status is CheckStatus.PASS
    assert DropoutRecurrenceCheck().category is CheckCategory.SUPPORTING


def test_dropout_recurrence_warns_when_dropouts_recur_across_windows():
    # 5/8 = 0.625 recent windows show a dropout, above the 0.3 default threshold.
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    hist = _hist("T008", "dropout_event_count", vals)
    r = DropoutRecurrenceCheck().run(_win(), {"history": hist})
    assert r.status is CheckStatus.WARNING
    assert _m(r, "recurrence_ratio") > 0.3


def test_dropout_recurrence_passes_with_insufficient_history():
    hist = _hist("T008", "dropout_event_count", [1.0, 1.0])
    r = DropoutRecurrenceCheck().run(_win(), {"history": hist})
    assert r.status is CheckStatus.PASS


def test_dropout_recurrence_passes_when_source_measurement_absent():
    hist = [{"T009": {"click_count": 5.0}}] * 8  # no T008.dropout_event_count
    r = DropoutRecurrenceCheck().run(_win(), {"history": hist})
    assert r.status is CheckStatus.PASS

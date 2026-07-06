from app.health.calibration import CalibrationProfile, MeasurementStats, compute_stats


def test_compute_stats_basic():
    stats = compute_stats([0.0, 1.0, 2.0, 3.0, 4.0])
    assert stats.count == 5
    assert stats.mean == 2.0
    assert stats.median == 2.0
    assert stats.minimum == 0.0
    assert stats.maximum == 4.0
    assert stats.p5 < stats.p95


def test_calibration_profile_defaults():
    p = CalibrationProfile(profile_id="x")
    assert p.profile_id == "x"
    assert p.version == 2
    assert p.sample_rate == 44100
    assert p.statistics == {}


import numpy as np  # noqa: E402

from app.health.calibration import generate_profile, iter_windows  # noqa: E402

SR = 44100


def _sine(seconds, freq=1000.0, amp=0.3):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_iter_windows_count_and_length():
    # 6 s signal, 2.5 s window, 0.5 s hop -> windows at 0,0.5,...,3.5 s = 8 windows.
    windows = list(iter_windows(_sine(6.0), SR, window_seconds=2.5, hop_seconds=0.5))
    assert len(windows) == 8
    assert all(w.shape[0] == int(2.5 * SR) for w in windows)


def test_iter_windows_too_short_yields_nothing():
    assert list(iter_windows(_sine(1.0), SR)) == []


def test_generate_profile_characterizes_all_checks():
    profile = generate_profile([_sine(6.0)], SR, profile_id="test")
    assert profile.window_count == 8
    assert profile.profile_id == "test"
    assert profile.created  # ISO date set
    assert set(profile.statistics) >= {
        "T001", "T002", "T003", "T004", "T005", "T006", "T007",
        "F001", "F002", "F003", "F004",
    }
    rms = profile.statistics["T002"]["rms"]
    assert rms.count == 8
    assert rms.mean > 0.0


from app.health.calibration import load_profile, save_profile  # noqa: E402


def test_save_load_round_trip(tmp_path):
    profile = generate_profile([_sine(6.0)], SR, profile_id="rt", sensor_info="piezo")
    path = tmp_path / "profile.json"
    save_profile(profile, str(path))
    loaded = load_profile(str(path))
    assert loaded.profile_id == "rt"
    assert loaded.sensor_info == "piezo"
    assert loaded.window_count == profile.window_count
    rms = loaded.statistics["T002"]["rms"]
    assert isinstance(rms, MeasurementStats)
    assert rms.count == profile.statistics["T002"]["rms"].count
    assert rms.mean == profile.statistics["T002"]["rms"].mean


import numpy as np  # noqa: E402 (already imported above, but harmless)
from app.health.calibration import (  # noqa: E402
    CalibrationProfile, generate_profile, save_profile, load_profile,
)
from app.health.models import (  # noqa: E402
    AudioWindow, Measurement, SignalCheckResult, CheckStatus, HealthReport, HealthState,
)


class _FakePipeline:
    """Emits two correlated measurements per window so covariance is non-diagonal."""
    def __init__(self):
        self._i = 0

    def analyze(self, window):
        self._i += 1
        a = float(self._i % 7)
        b = a * 2.0 + 1.0  # perfectly correlated with a
        res = SignalCheckResult(
            check_id="T002", check_name="rms", status=CheckStatus.PASS, executed=True,
            measurements=[Measurement("rms", a), Measurement("peak", b)],
        )
        return HealthReport(timestamp=0.0, window_id="x", check_results=[res],
                            final_state=HealthState.OK)


def test_generate_profile_stores_mean_and_covariance():
    sig = np.zeros(int(44100 * 6.0), dtype=np.float32)  # 6 s -> several windows
    profile = generate_profile([sig], 44100, profile_id="test_v2", pipeline=_FakePipeline())
    assert profile.version == 2
    assert [tuple(k) for k in profile.feature_index] == [("T002", "peak"), ("T002", "rms")] \
        or [tuple(k) for k in profile.feature_index] == [("T002", "rms"), ("T002", "peak")]
    D = len(profile.feature_index)
    assert len(profile.mean_vector) == D
    assert len(profile.covariance) == D and all(len(row) == D for row in profile.covariance)


def test_profile_v2_round_trips(tmp_path):
    sig = np.zeros(int(44100 * 6.0), dtype=np.float32)
    profile = generate_profile([sig], 44100, profile_id="test_v2", pipeline=_FakePipeline())
    path = tmp_path / "p.json"
    save_profile(profile, str(path))
    loaded = load_profile(str(path))
    assert loaded.version == 2
    assert loaded.feature_index == profile.feature_index
    assert loaded.mean_vector == profile.mean_vector
    assert loaded.covariance == profile.covariance
    # statistics (used by calibration_eval) still present
    assert "T002" in loaded.statistics

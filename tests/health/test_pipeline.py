import numpy as np

from app.health.checks.base import SignalHealthCheck
from app.health.manager import SignalCheckManager
from app.health.models import AudioWindow, CheckStatus, HealthState, SignalCheckResult
from app.health.pipeline import HealthAnalysisPipeline


def _window():
    return AudioWindow(samples=np.zeros(110250, dtype=np.float32), sample_rate=44100)


class _PassingCheck(SignalHealthCheck):
    check_id = "T999"
    check_name = "Passing"

    def run(self, window, features):
        return SignalCheckResult(
            check_id=self.check_id, check_name=self.check_name, status=CheckStatus.PASS
        )


def test_pipeline_with_no_checks_is_unknown():
    report = HealthAnalysisPipeline().analyze(_window())
    assert report.final_state is HealthState.UNKNOWN
    assert report.check_results == []
    assert report.confidence == 0.0
    assert report.window_id  # non-empty id
    assert report.timestamp > 0
    assert report.diagnostic_summary


def test_pipeline_runs_registered_checks():
    manager = SignalCheckManager()
    manager.register(_PassingCheck())
    report = HealthAnalysisPipeline(manager=manager).analyze(_window())
    assert len(report.check_results) == 1
    assert report.check_results[0].check_id == "T999"
    # The manager's post-run mutation propagates through the pipeline.
    assert report.check_results[0].executed is True
    assert report.check_results[0].execution_time >= 0.0


def test_pipeline_produces_unique_window_ids():
    pipeline = HealthAnalysisPipeline()
    id1 = pipeline.analyze(_window()).window_id
    id2 = pipeline.analyze(_window()).window_id
    assert id1 != id2


from app.health.calibration import generate_profile  # noqa: E402
from app.health.calibration_eval import CalibrationEvaluation  # noqa: E402
from app.health.config import pipeline_for_profile  # noqa: E402

SR_CAL = 44100


def _sine6():
    n = int(6.0 * SR_CAL)
    return (0.3 * np.sin(2 * np.pi * 1000.0 * np.arange(n) / SR_CAL)).astype(np.float32)


def test_pipeline_without_profile_has_no_calibration_eval():
    window = AudioWindow(samples=np.zeros(110250, dtype=np.float32), sample_rate=SR_CAL)
    report = pipeline_for_profile("development").analyze(window)
    assert report.calibration_evaluation is None


def test_pipeline_with_profile_produces_calibration_eval():
    profile = generate_profile([_sine6()], SR_CAL, profile_id="p")
    sine_window = AudioWindow(samples=_sine6()[:110250], sample_rate=SR_CAL)
    report = pipeline_for_profile("development", calibration_profile=profile).analyze(sine_window)
    assert isinstance(report.calibration_evaluation, CalibrationEvaluation)


from app.health.models import Measurement  # noqa: E402


def test_pipeline_history_accumulates_and_passes_prior_windows():
    seen = []

    class _HistorySpy(SignalHealthCheck):
        check_id = "SPY"
        check_name = "spy"

        def run(self, window, features):
            seen.append(len(features.get("history", [])))
            return SignalCheckResult(
                check_id="SPY", check_name="spy", measurements=[Measurement("v", 1.0)]
            )

    manager = SignalCheckManager()
    manager.register(_HistorySpy())
    pipeline = HealthAnalysisPipeline(manager=manager, history_length=3)
    for _ in range(5):
        pipeline.analyze(_window())
    # Each run sees prior windows only, bounded by history_length=3.
    assert seen == [0, 1, 2, 3, 3]


def test_pipeline_populates_anomaly_and_confidence_with_profile():
    import numpy as np
    from app.health.calibration import generate_profile
    from app.health.config import pipeline_for_profile

    sr, n = 44100, int(44100 * 2.5)
    rng = np.random.default_rng(0)
    healthy = [(0.2 * np.sin(2 * np.pi * 1000 * np.arange(n) / sr)
                + 0.01 * rng.standard_normal(n)).astype(np.float32) for _ in range(3)]
    profile = generate_profile(healthy, sr, profile_id="t")

    pipe = pipeline_for_profile("development", calibration_profile=profile)
    report = pipe.analyze(AudioWindow(samples=healthy[0], sample_rate=sr))
    assert report.anomaly_result is not None
    assert report.confidence == report.anomaly_result.confidence


def test_pipeline_without_profile_has_no_anomaly():
    from app.health.config import pipeline_for_profile

    pipe = pipeline_for_profile("development")  # no calibration profile
    report = pipe.analyze(_window())
    assert report.anomaly_result is None

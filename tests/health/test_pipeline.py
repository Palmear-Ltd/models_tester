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

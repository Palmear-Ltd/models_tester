import numpy as np
import pytest

from app.health.checks.base import SignalHealthCheck
from app.health.manager import SignalCheckManager
from app.health.models import AudioWindow, CheckStatus, SignalCheckResult


def _window():
    return AudioWindow(samples=np.zeros(100, dtype=np.float32), sample_rate=44100)


class _PassingCheck(SignalHealthCheck):
    check_id = "T999"
    check_name = "Passing"

    def run(self, window, features):
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
        )


class _ExplodingCheck(SignalHealthCheck):
    check_id = "T998"
    check_name = "Exploding"

    def run(self, window, features):
        raise RuntimeError("boom")


def test_register_and_run_check():
    manager = SignalCheckManager()
    manager.register(_PassingCheck())
    results = manager.run_checks(_window(), {})
    assert len(results) == 1
    assert results[0].check_id == "T999"
    assert results[0].executed is True
    assert results[0].status is CheckStatus.PASS
    assert results[0].execution_time >= 0.0


def test_failing_check_is_isolated():
    manager = SignalCheckManager()
    manager.register(_ExplodingCheck())
    manager.register(_PassingCheck())
    results = manager.run_checks(_window(), {})
    assert len(results) == 2  # one failure does not stop the others
    failed = results[0]
    assert failed.check_id == "T998"
    assert failed.executed is False
    assert failed.status is CheckStatus.NOT_EXECUTED
    assert any("boom" in msg for msg in failed.diagnostic_messages)
    # Timing is always recorded, even on the failure path.
    assert failed.execution_time >= 0.0
    assert results[1].status is CheckStatus.PASS


class _NoIdCheck(SignalHealthCheck):
    check_name = "Missing id"

    def run(self, window, features):  # pragma: no cover - never registered
        return SignalCheckResult(check_id="", check_name=self.check_name)


def test_register_rejects_check_without_id():
    manager = SignalCheckManager()
    with pytest.raises(ValueError):
        manager.register(_NoIdCheck())


def test_checks_property_returns_copy():
    manager = SignalCheckManager()
    manager.register(_PassingCheck())
    checks = manager.checks
    checks.clear()
    assert len(manager.checks) == 1

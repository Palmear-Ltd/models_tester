"""Signal Check Manager: registry and isolated execution of Signal Health Checks."""
from __future__ import annotations

import time
from typing import Any

from app.health.checks.base import SignalHealthCheck
from app.health.models import AudioWindow, CheckStatus, SignalCheckResult


class SignalCheckManager:
    """Holds the registered checks and runs them, isolating individual failures."""

    def __init__(self):
        self._checks: list[SignalHealthCheck] = []

    def register(self, check: SignalHealthCheck) -> None:
        # Fail fast on misconfiguration: a blank check_id makes result correlation
        # ambiguous (the manager would fall back to the class name in diagnostics).
        if not check.check_id:
            raise ValueError(
                f"{check.__class__.__name__} must set a non-empty check_id before registration"
            )
        self._checks.append(check)

    @property
    def checks(self) -> list[SignalHealthCheck]:
        return list(self._checks)

    def run_checks(
        self, window: AudioWindow, features: dict[str, Any]
    ) -> list[SignalCheckResult]:
        results: list[SignalCheckResult] = []
        for check in self._checks:
            start = time.perf_counter()
            try:
                result = check.run(window, features)
                result.executed = True
            except Exception as exc:  # isolate one check's failure from the rest
                result = SignalCheckResult(
                    check_id=getattr(check, "check_id", "") or check.__class__.__name__,
                    check_name=getattr(check, "check_name", "")
                    or check.__class__.__name__,
                    status=CheckStatus.NOT_EXECUTED,
                    executed=False,
                    diagnostic_messages=[f"Check raised: {exc}"],
                )
            result.execution_time = time.perf_counter() - start
            results.append(result)
        return results

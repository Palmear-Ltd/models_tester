"""UI-agnostic formatting of a HealthReport into display rows (spec §11).

Pure functions returning plain Python types — no Tkinter/UI imports — so the
health package stays portable. The tester's panel renders these rows; any other
front end can reuse them.
"""
from __future__ import annotations

from app.health.models import HealthReport, Measurement, SignalCheckResult


def _format_value(value: float) -> str:
    return f"{value:.4g}"


def _format_measurements(measurements: list[Measurement]) -> str:
    parts = []
    for m in measurements:
        unit = f" {m.unit}" if m.unit else ""
        parts.append(f"{m.name}={_format_value(m.value)}{unit}")
    return ", ".join(parts)


def check_row(result: SignalCheckResult) -> tuple[str, str, str, str]:
    """Return (check_id, check_name, status, detail) for one check result.

    `detail` is the diagnostic message(s) when the check is not PASS, otherwise a
    compact list of its measurements.
    """
    if result.diagnostic_messages:
        detail = "; ".join(result.diagnostic_messages)
    else:
        detail = _format_measurements(result.measurements)
    return (result.check_id, result.check_name, result.status.value, detail)


def report_rows(report: HealthReport) -> list[tuple[str, str, str, str]]:
    """One display row per check result, in pipeline execution order."""
    return [check_row(r) for r in report.check_results]

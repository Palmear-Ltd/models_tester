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


def root_cause_row(assessment) -> tuple[str, str, str, str, str]:
    """Return a synthetic 5-tuple row summarizing a root-cause assessment.

    Shares the same shape as one row of `report_rows` — (check_id, name,
    status, detail, cal) — so the UI can insert it into the same tree using
    the same insert call. `status` holds the primary cause's `.value` and
    `detail` holds the plain-language explanation; `cal` is unused (empty).
    """
    return ("CAUSE", "Likely Cause", assessment.primary_cause.value, assessment.explanation, "")


def report_rows(report: HealthReport) -> list[tuple[str, str, str, str, str]]:
    """One display row per check: (check_id, name, status, detail, calibration).

    `calibration` is the worst calibration verdict for that check from
    `report.calibration_evaluation` ("FAIL" beats "WARNING"), or "" when the
    check has no calibration deviation (or no profile is loaded).
    """
    cal_by_check: dict[str, str] = {}
    evaluation = report.calibration_evaluation
    if evaluation is not None:
        for d in evaluation.deviations:
            verdict = d.verdict.value  # "FAIL" or "WARNING"
            if verdict == "FAIL" or d.check_id not in cal_by_check:
                cal_by_check[d.check_id] = verdict
    rows = []
    for r in report.check_results:
        check_id, name, status, detail = check_row(r)
        rows.append((check_id, name, status, detail, cal_by_check.get(check_id, "")))
    return rows

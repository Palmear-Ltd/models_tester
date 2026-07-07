from app.health.models import (
    CheckStatus,
    HealthReport,
    Measurement,
    SignalCheckResult,
)
from app.health.reporting import check_row, report_rows, root_cause_row
from app.health.rootcause import RootCause, RootCauseAssessment


def _result(cid, name, status, measurements=None, diag=None):
    return SignalCheckResult(
        check_id=cid,
        check_name=name,
        status=status,
        measurements=measurements or [],
        diagnostic_messages=diag or [],
    )


def test_check_row_shows_measurements_when_passing():
    r = _result(
        "T002", "Signal Energy", CheckStatus.PASS,
        measurements=[Measurement("rms", 0.21034)],
    )
    cid, name, status, detail = check_row(r)
    assert cid == "T002"
    assert name == "Signal Energy"
    assert status == "PASS"
    assert "rms=0.2103" in detail


def test_check_row_shows_diagnostics_when_not_passing():
    r = _result(
        "T001", "Flatline Detection", CheckStatus.FAIL,
        measurements=[Measurement("std", 0.0)],
        diag=["Flatline: dead signal"],
    )
    _, _, status, detail = check_row(r)
    assert status == "FAIL"
    assert detail == "Flatline: dead signal"


def test_measurement_unit_is_included():
    r = _result(
        "F001", "Spectral Shape", CheckStatus.PASS,
        measurements=[Measurement("spectral_centroid", 1002.0, "Hz")],
    )
    _, _, _, detail = check_row(r)
    assert "spectral_centroid=1002 Hz" in detail


def test_report_rows_one_per_check():
    report = HealthReport(
        timestamp=0.0,
        window_id="w",
        check_results=[
            _result("T001", "Flatline Detection", CheckStatus.PASS),
            _result("T002", "Signal Energy", CheckStatus.PASS),
        ],
    )
    rows = report_rows(report)
    assert len(rows) == 2
    assert rows[0][0] == "T001"
    assert rows[1][0] == "T002"


from app.health.calibration_eval import CalibrationEvaluation, MeasurementDeviation  # noqa: E402


def test_report_rows_blank_calibration_when_none():
    report = HealthReport(
        timestamp=0.0, window_id="w",
        check_results=[_result("T001", "Flatline", CheckStatus.PASS)],
    )
    rows = report_rows(report)
    assert rows[0][4] == ""  # 5th element = calibration verdict


def test_report_rows_marks_calibration_deviation_per_check():
    ev = CalibrationEvaluation(
        deviations=[MeasurementDeviation("T002", "rms", 9.9, 0.1, 0.3, CheckStatus.FAIL)],
        warn_count=0, fault_count=1,
    )
    report = HealthReport(
        timestamp=0.0, window_id="w",
        check_results=[
            _result("T001", "Flatline", CheckStatus.PASS),
            _result("T002", "Signal Energy", CheckStatus.PASS),
        ],
        calibration_evaluation=ev,
    )
    by_id = {r[0]: r for r in report_rows(report)}
    assert by_id["T002"][4] == "FAIL"
    assert by_id["T001"][4] == ""


def test_root_cause_row_shape_and_content():
    assessment = RootCauseAssessment(
        primary_cause=RootCause.CABLE,
        confidence=0.5,
        explanation="Likely a cable problem: there was a complete loss of signal (flatline).",
        ranked_causes=[(RootCause.CABLE, 4.0, "reason")],
        contributing_check_ids=["T001"],
    )
    row = root_cause_row(assessment)
    assert row == (
        "CAUSE",
        "Likely Cause",
        "CABLE",
        "Likely a cable problem: there was a complete loss of signal (flatline).",
        "",
    )

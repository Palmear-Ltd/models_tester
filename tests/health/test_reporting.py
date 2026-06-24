from app.health.models import (
    CheckStatus,
    HealthReport,
    Measurement,
    SignalCheckResult,
)
from app.health.reporting import check_row, report_rows


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

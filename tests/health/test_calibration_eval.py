from app.health.calibration import CalibrationProfile, MeasurementStats
from app.health.calibration_eval import CalibrationEvaluation, evaluate_calibration
from app.health.models import CheckStatus, Measurement, SignalCheckResult


def _stats(p5, p95, mn, mx):
    return MeasurementStats(
        count=100, mean=(p5 + p95) / 2, median=(p5 + p95) / 2, std=1.0,
        minimum=mn, maximum=mx, p5=p5, p95=p95,
    )


def _profile():
    return CalibrationProfile(
        profile_id="p", statistics={"T002": {"rms": _stats(0.1, 0.3, 0.05, 0.4)}}
    )


def _result(check_id, **meas):
    return SignalCheckResult(
        check_id=check_id,
        check_name=check_id,
        measurements=[Measurement(n, v) for n, v in meas.items()],
    )


def test_pass_inside_band():
    ev = evaluate_calibration([_result("T002", rms=0.2)], _profile())
    assert ev.deviations == []
    assert ev.warn_count == 0 and ev.fault_count == 0


def test_warning_outside_band_within_range():
    # 0.35 > p95 (0.3) but < fault_hi (0.4 + 0.5*0.35 = 0.575) -> WARNING.
    ev = evaluate_calibration([_result("T002", rms=0.35)], _profile())
    assert ev.warn_count == 1 and ev.fault_count == 0
    assert ev.deviations[0].verdict is CheckStatus.WARNING


def test_fault_far_outside_range():
    ev = evaluate_calibration([_result("T002", rms=2.0)], _profile())
    assert ev.fault_count == 1
    assert ev.deviations[0].verdict is CheckStatus.FAIL


def test_unprofiled_measurements_and_checks_are_skipped():
    ev = evaluate_calibration(
        [_result("T002", rms=0.2, extra=99.0), _result("T999", foo=1.0)], _profile()
    )
    assert ev.deviations == []  # rms passes; extra/foo/T999 not in profile


def test_degenerate_constant_measurement():
    prof = CalibrationProfile(
        profile_id="p", statistics={"T006": {"dc_offset": _stats(0.0, 0.0, 0.0, 0.0)}}
    )
    assert evaluate_calibration([_result("T006", dc_offset=0.0)], prof).deviations == []
    assert evaluate_calibration([_result("T006", dc_offset=0.5)], prof).fault_count == 1

import math

import pytest

from app.decision.sprt import (
    SPRTAccumulator,
    SPRTConfig,
    default_config,
    llr_increment,
    sprt_boundaries,
)


def test_boundaries_golden_values():
    a, b = sprt_boundaries(0.05, 0.05)
    assert a == pytest.approx(math.log(19), rel=1e-9)
    assert b == pytest.approx(-math.log(19), rel=1e-9)


def test_boundaries_rejects_out_of_range_alpha_beta():
    with pytest.raises(ValueError):
        sprt_boundaries(0.0, 0.05)
    with pytest.raises(ValueError):
        sprt_boundaries(0.05, 1.0)


def test_boundaries_monotonic_in_alpha():
    a1, _ = sprt_boundaries(0.10, 0.05)
    a2, _ = sprt_boundaries(0.01, 0.05)
    assert a2 > a1  # smaller alpha (stricter FP tolerance) requires more evidence


def test_all_high_scores_decides_infested_within_bounded_windows():
    acc = SPRTAccumulator(default_config(alpha=0.05, beta=0.05, weight=1.0))
    for _ in range(50):
        step = acc.update(0.95)
        if step.state != "CONTINUE":
            break
    result = acc.result()
    assert result.final_state == "INFESTED"
    assert result.decided_at_index is not None
    assert result.decided_at_index < 50


def test_all_low_scores_decides_healthy_within_bounded_windows():
    acc = SPRTAccumulator(default_config(alpha=0.05, beta=0.05, weight=1.0))
    for _ in range(50):
        step = acc.update(0.05)
        if step.state != "CONTINUE":
            break
    result = acc.result()
    assert result.final_state == "HEALTHY"
    assert result.decided_at_index is not None
    assert result.decided_at_index < 50


def test_noisy_but_healthy_does_not_false_positive():
    # Regression guard for the exact complaint that motivated this work: a noisy but truly
    # healthy session (scores jittering around ~0.3, well below the infested class) must
    # not accumulate enough evidence to cross into INFESTED over a realistic session length.
    import random

    rng = random.Random(1234)
    acc = SPRTAccumulator(default_config(alpha=0.05, beta=0.05, weight=0.2))
    for _ in range(300):  # 300 windows * 0.5s hop = 150s session
        score = min(max(rng.gauss(0.30, 0.08), 0.0), 1.0)
        acc.update(score)
    result = acc.result()
    assert result.final_state != "INFESTED"


def test_borderline_oscillating_stays_suspicious():
    acc = SPRTAccumulator(default_config(alpha=0.05, beta=0.05, weight=0.2))
    for i in range(120):  # 60s session
        score = 0.45 if i % 2 == 0 else 0.55
        acc.update(score)
    result = acc.result()
    assert result.final_state == "SUSPICIOUS"
    assert result.decided_at_index is None


def test_eps_clipping_avoids_inf_and_nan():
    config = default_config()
    inc_zero = llr_increment(0.0, config)
    inc_one = llr_increment(1.0, config)
    assert math.isfinite(inc_zero)
    assert math.isfinite(inc_one)
    assert inc_zero < 0.0
    assert inc_one > 0.0


def test_accumulator_latches_state_after_crossing():
    acc = SPRTAccumulator(default_config(alpha=0.05, beta=0.05, weight=1.0))
    decided_index = None
    for i in range(40):
        step = acc.update(0.95)
        if step.state == "INFESTED" and decided_index is None:
            decided_index = i
    assert decided_index is not None
    # Feed contradicting evidence after the decision — state must not flip back.
    for _ in range(20):
        step = acc.update(0.01)
    assert step.state == "INFESTED"
    result = acc.result()
    assert result.final_state == "INFESTED"
    assert result.decided_at_index == decided_index


def test_fitted_mode_requires_fitted_params():
    with pytest.raises(ValueError):
        SPRTConfig(likelihood_mode="fitted", fitted_params=None)


def test_default_config_falls_back_to_logit_when_no_calibration_file(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.json")
    config = default_config(likelihood_path=missing_path)
    assert config.likelihood_mode == "logit"

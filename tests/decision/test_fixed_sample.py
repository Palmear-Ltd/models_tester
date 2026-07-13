import random

import pytest

from app.decision.fixed_sample import classify, fit_quantile_thresholds


def _samples(mean, spread, n, seed):
    rng = random.Random(seed)
    return [min(max(rng.gauss(mean, spread), 0.0), 1.0) for _ in range(n)]


def test_fit_quantile_thresholds_separated_classes_collapses_to_single_cutoff():
    healthy = _samples(0.15, 0.05, 500, seed=1)
    infested = _samples(0.85, 0.05, 500, seed=2)
    thresholds = fit_quantile_thresholds(healthy, infested, alpha=0.05, beta=0.05)
    # Well-separated classes: no ambiguous band needed.
    assert thresholds.t_low == thresholds.t_high


def test_fit_quantile_thresholds_overlapping_classes_has_a_band():
    healthy = _samples(0.45, 0.15, 500, seed=3)
    infested = _samples(0.55, 0.15, 500, seed=4)
    thresholds = fit_quantile_thresholds(healthy, infested, alpha=0.05, beta=0.05)
    assert thresholds.t_low < thresholds.t_high


def test_fit_quantile_thresholds_achieves_target_fpr_on_held_out_healthy():
    # Overlapping (not well-separated) classes, so the fitted band is non-degenerate and
    # the quantile-defined cutoff — not a collapsed midpoint — is what's under test.
    healthy_fit = _samples(0.45, 0.15, 3000, seed=5)
    infested_fit = _samples(0.55, 0.15, 3000, seed=6)
    thresholds = fit_quantile_thresholds(healthy_fit, infested_fit, alpha=0.05, beta=0.05)
    assert thresholds.t_low < thresholds.t_high

    healthy_eval = _samples(0.45, 0.15, 5000, seed=7)
    false_positive_rate = sum(1 for s in healthy_eval if classify(s, thresholds) == "INFESTED") / len(healthy_eval)
    assert false_positive_rate == pytest.approx(0.05, abs=0.02)


def test_fit_quantile_thresholds_achieves_target_fnr_on_held_out_infested():
    healthy_fit = _samples(0.45, 0.15, 3000, seed=8)
    infested_fit = _samples(0.55, 0.15, 3000, seed=9)
    thresholds = fit_quantile_thresholds(healthy_fit, infested_fit, alpha=0.05, beta=0.05)
    assert thresholds.t_low < thresholds.t_high

    infested_eval = _samples(0.55, 0.15, 5000, seed=10)
    false_negative_rate = sum(1 for s in infested_eval if classify(s, thresholds) == "HEALTHY") / len(infested_eval)
    assert false_negative_rate == pytest.approx(0.05, abs=0.02)


def test_fit_quantile_thresholds_requires_minimum_samples():
    with pytest.raises(ValueError):
        fit_quantile_thresholds([0.1], [0.9, 0.8], alpha=0.05, beta=0.05)
    with pytest.raises(ValueError):
        fit_quantile_thresholds([0.1, 0.2], [0.9], alpha=0.05, beta=0.05)


def test_fit_quantile_thresholds_rejects_out_of_range_alpha_beta():
    healthy = [0.1, 0.2, 0.15]
    infested = [0.8, 0.9, 0.85]
    with pytest.raises(ValueError):
        fit_quantile_thresholds(healthy, infested, alpha=0.0, beta=0.05)
    with pytest.raises(ValueError):
        fit_quantile_thresholds(healthy, infested, alpha=0.05, beta=1.0)


def test_classify_boundaries():
    from app.decision.fixed_sample import QuantileThresholds

    thresholds = QuantileThresholds(t_low=0.3, t_high=0.7, alpha=0.05, beta=0.05)
    assert classify(0.71, thresholds) == "INFESTED"
    assert classify(0.29, thresholds) == "HEALTHY"
    assert classify(0.5, thresholds) == "SUSPICIOUS"

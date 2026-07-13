import random

import pytest

from app.decision.likelihood import (
    FittedLikelihood,
    fit_beta_params,
    fitted_llr_increment,
)


def _sample_beta(a, b, n, seed):
    rng = random.Random(seed)
    return [rng.betavariate(a, b) for _ in range(n)]


def test_fit_beta_params_recovers_known_shape_healthy_like():
    true_a, true_b = 2.0, 8.0  # mass concentrated near 0, like a healthy-class score
    samples = _sample_beta(true_a, true_b, 5000, seed=1)
    a, b = fit_beta_params(samples)
    assert a == pytest.approx(true_a, rel=0.15)
    assert b == pytest.approx(true_b, rel=0.15)


def test_fit_beta_params_recovers_known_shape_infested_like():
    true_a, true_b = 8.0, 2.0  # mass concentrated near 1, like an infested-class score
    samples = _sample_beta(true_a, true_b, 5000, seed=2)
    a, b = fit_beta_params(samples)
    assert a == pytest.approx(true_a, rel=0.15)
    assert b == pytest.approx(true_b, rel=0.15)


def test_fit_beta_params_handles_near_constant_scores():
    samples = [0.5 + 1e-9 * i for i in range(10)]
    a, b = fit_beta_params(samples)
    assert a > 0
    assert b > 0


def test_fit_beta_params_requires_at_least_two_scores():
    with pytest.raises(ValueError):
        fit_beta_params([0.5])


def test_fitted_llr_increment_sign_near_each_class_mode():
    fitted = FittedLikelihood(healthy_beta=(2.0, 8.0), infested_beta=(8.0, 2.0))
    # Near the healthy mode (low score): evidence should favor healthy (negative LLR).
    assert fitted_llr_increment(0.1, fitted) < 0.0
    # Near the infested mode (high score): evidence should favor infested (positive LLR).
    assert fitted_llr_increment(0.9, fitted) > 0.0


def test_fitted_likelihood_json_roundtrip():
    fitted = FittedLikelihood(healthy_beta=(2.0, 8.0), infested_beta=(8.0, 2.0))
    restored = FittedLikelihood.from_json(fitted.to_json())
    assert restored.healthy_beta == fitted.healthy_beta
    assert restored.infested_beta == fitted.infested_beta

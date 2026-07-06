import math
from app.health.chi2 import chi2_ppf, chi2_cdf


def test_ppf_matches_known_values():
    assert math.isclose(chi2_ppf(0.95, 1), 3.8415, abs_tol=1e-2)
    assert math.isclose(chi2_ppf(0.95, 2), 5.9915, abs_tol=1e-2)
    assert math.isclose(chi2_ppf(0.999, 1), 10.828, abs_tol=1e-2)
    assert math.isclose(chi2_ppf(0.95, 10), 18.307, abs_tol=1e-2)


def test_cdf_ppf_roundtrip():
    for df in (1, 3, 8, 14):
        x = chi2_ppf(0.9, df)
        assert math.isclose(chi2_cdf(x, df), 0.9, abs_tol=1e-4)


def test_cdf_monotonic_and_bounded():
    assert chi2_cdf(0.0, 5) == 0.0
    assert 0.0 <= chi2_cdf(1.0, 5) <= chi2_cdf(20.0, 5) <= 1.0

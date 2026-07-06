"""Chi-square CDF and inverse CDF in pure stdlib math (no SciPy/NumPy).

chi2_cdf(x, df) = P(df/2, x/2), the regularized lower incomplete gamma.
chi2_ppf inverts it by bisection. Adequate precision for anomaly thresholds.
"""
from __future__ import annotations

import math

_MAXIT = 300
_EPS = 1e-14
_FPMIN = 1e-300


def _gammap(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        # Series representation.
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(_MAXIT):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * _EPS:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for Q(a, x) = 1 - P(a, x).
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAXIT):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_cdf(x: float, df: int) -> float:
    if x <= 0.0:
        return 0.0
    return _gammap(df / 2.0, x / 2.0)


def chi2_ppf(p: float, df: int) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return math.inf
    lo, hi = 0.0, max(1.0, float(df))
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

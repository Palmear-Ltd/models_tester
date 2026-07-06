import numpy as np
from app.health.anomaly import AnomalyResult, detect_anomaly


class _M:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _R:
    def __init__(self, check_id, measurements):
        self.check_id = check_id
        self.measurements = measurements


class _Profile:
    def __init__(self, feature_index, mean_vector, covariance):
        self.feature_index = feature_index
        self.mean_vector = mean_vector
        self.covariance = covariance


def _profile():
    # Two independent features, unit-ish variance.
    return _Profile(
        feature_index=[["T002", "rms"], ["F001", "spectral_centroid"]],
        mean_vector=[0.2, 1000.0],
        covariance=[[0.0025, 0.0], [0.0, 10000.0]],  # std 0.05 and 100
    )


def test_returns_none_on_dimension_mismatch():
    # A malformed profile (mean vector shorter than the feature index) must be
    # rejected as a no-op, not raise IndexError into the audio path.
    profile = _Profile(
        feature_index=[["C", "a"], ["C", "b"]],
        mean_vector=[0.0],                      # length 1, but D=2
        covariance=[[1.0, 0.0], [0.0, 1.0]],
    )
    results = [_R("C", [_M("a", 1.0), _M("b", 1.0)])]
    assert detect_anomaly(results, profile) is None


def test_at_mean_is_not_anomalous():
    results = [_R("T002", [_M("rms", 0.2)]), _R("F001", [_M("spectral_centroid", 1000.0)])]
    r = detect_anomaly(results, _profile())
    assert isinstance(r, AnomalyResult)
    assert r.distance < 1e-6
    assert r.is_anomalous is False
    assert abs(r.confidence - 1.0) < 1e-6


def test_far_out_is_anomalous():
    # rms 12 std out, centroid 1 std out -> d^2 = 144 + 1 = 145, d ~ 12.04
    results = [_R("T002", [_M("rms", 0.8)]), _R("F001", [_M("spectral_centroid", 1100.0)])]
    r = detect_anomaly(results, _profile(), p=0.001)
    assert r.is_anomalous is True
    assert r.distance > r.threshold
    assert r.confidence == 0.0
    assert r.contributors[0][0] == "T002.rms"  # dominant contributor


def test_correlation_caught_that_diagonal_would_miss():
    # Two strongly correlated features; a point that moves against the correlation
    # is far in Mahalanobis distance even though each marginal z is modest.
    profile = _Profile(
        feature_index=[["C", "a"], ["C", "b"]],
        mean_vector=[0.0, 0.0],
        covariance=[[1.0, 0.9], [0.9, 1.0]],
    )
    results = [_R("C", [_M("a", 2.0), _M("b", -2.0)])]  # opposes the +0.9 correlation
    r = detect_anomaly(profile=profile, results=results, p=0.05)
    # Diagonal z-distance would be sqrt((4+4)/2)=2.0; full Mahalanobis is much larger.
    assert r.distance > 4.0
    assert r.is_anomalous is True


def test_returns_none_without_covariance():
    profile = _Profile(feature_index=[], mean_vector=[], covariance=[])
    results = [_R("T002", [_M("rms", 0.2)])]
    assert detect_anomaly(results, profile) is None


def test_singular_covariance_is_regularized_not_crash():
    profile = _Profile(
        feature_index=[["C", "a"], ["C", "b"]],
        mean_vector=[0.0, 0.0],
        covariance=[[1.0, 1.0], [1.0, 1.0]],  # singular
    )
    results = [_R("C", [_M("a", 1.0), _M("b", 1.0)])]
    r = detect_anomaly(results, profile)
    assert isinstance(r, AnomalyResult)
    assert np.isfinite(r.distance)


def test_missing_feature_uses_submatrix():
    # Only one of the two profile features is present this window.
    results = [_R("T002", [_M("rms", 0.2)])]  # no spectral_centroid
    r = detect_anomaly(results, _profile())
    assert isinstance(r, AnomalyResult)
    assert r.distance < 1e-6  # rms at its mean -> zero distance over the 1-D submatrix

from app.health.anomaly import AnomalyResult, detect_anomaly


class _Stat:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class _Profile:
    def __init__(self, statistics):
        self.statistics = statistics


class _M:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _R:
    def __init__(self, check_id, measurements):
        self.check_id = check_id
        self.measurements = measurements


def _profile():
    return _Profile({"T002": {"rms": _Stat(mean=0.2, std=0.05)},
                     "F001": {"spectral_centroid": _Stat(mean=1000.0, std=100.0)}})


def test_at_mean_is_not_anomalous():
    results = [_R("T002", [_M("rms", 0.2)]), _R("F001", [_M("spectral_centroid", 1000.0)])]
    r = detect_anomaly(results, _profile())
    assert isinstance(r, AnomalyResult)
    assert r.distance == 0.0
    assert r.is_anomalous is False
    assert r.confidence == 1.0


def test_far_out_is_anomalous_with_low_confidence():
    # rms 12 sigma out, centroid 1 sigma out -> RMS z = sqrt((144+1)/2) ~ 8.5,
    # which is >= 2*threshold (6.0) so confidence floors at 0.0.
    results = [_R("T002", [_M("rms", 0.8)]), _R("F001", [_M("spectral_centroid", 1100.0)])]
    r = detect_anomaly(results, _profile(), threshold=3.0)
    assert r.is_anomalous is True
    assert r.distance > 3.0
    assert r.confidence == 0.0
    assert r.contributors[0][0] == "T002.rms"  # dominant deviation


def test_returns_none_when_no_measurements_match_profile():
    results = [_R("Z999", [_M("foo", 1.0)])]
    assert detect_anomaly(results, _profile()) is None


def test_skips_zero_std_measurements():
    profile = _Profile({"T002": {"rms": _Stat(mean=0.2, std=0.0)}})
    results = [_R("T002", [_M("rms", 5.0)])]
    assert detect_anomaly(results, profile) is None  # only stat skipped -> no usable z


def test_confidence_is_half_at_threshold():
    # one measurement exactly `threshold` sigma out -> distance == threshold
    profile = _Profile({"T002": {"rms": _Stat(mean=0.0, std=1.0)}})
    results = [_R("T002", [_M("rms", 3.0)])]
    r = detect_anomaly(results, profile, threshold=3.0)
    assert r.distance == 3.0
    assert r.confidence == 0.5

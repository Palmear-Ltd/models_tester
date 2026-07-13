from app.decision.baselines import current_method, ewma_peak, mean_threshold, median_threshold


def test_mean_threshold_healthy():
    result = mean_threshold([0.1, 0.2, 0.15, 0.05], threshold=0.5)
    assert result.final_state == "HEALTHY"
    assert result.statistic == 0.125


def test_mean_threshold_infested():
    result = mean_threshold([0.9, 0.85, 0.95, 0.99], threshold=0.5)
    assert result.final_state == "INFESTED"


def test_mean_threshold_empty_scores_is_healthy():
    result = mean_threshold([], threshold=0.5)
    assert result.final_state == "HEALTHY"
    assert result.statistic == 0.0


def test_median_threshold_odd_and_even_counts():
    odd = median_threshold([0.1, 0.9, 0.2], threshold=0.5)
    assert odd.statistic == 0.2
    even = median_threshold([0.1, 0.9, 0.2, 0.8], threshold=0.5)
    assert even.statistic == 0.5


def test_ewma_peak_tracks_a_late_spike():
    scores = [0.05] * 20 + [0.99] * 20
    result = ewma_peak(scores, threshold=0.5, span=5.0)
    assert result.final_state == "INFESTED"
    assert result.statistic > 0.5


def test_ewma_peak_noisy_healthy_stays_below_threshold():
    scores = [0.2, 0.3, 0.25, 0.35, 0.28, 0.22] * 10
    result = ewma_peak(scores, threshold=0.5, span=5.0)
    assert result.final_state == "HEALTHY"


def test_ewma_peak_empty_scores_is_healthy():
    result = ewma_peak([], threshold=0.5)
    assert result.final_state == "HEALTHY"
    assert result.statistic == 0.0


def test_current_method_reproduces_main_py_bands():
    # 20 positive events out of some window count, thresh=0.5, susp=17, inf=27
    scores = [0.9] * 20 + [0.1] * 10
    assert current_method(scores, score_thresh=0.5, susp_limit=17, inf_limit=27) == "SUSPICIOUS"
    scores_low = [0.9] * 5 + [0.1] * 25
    assert current_method(scores_low, score_thresh=0.5, susp_limit=17, inf_limit=27) == "HEALTHY"
    scores_high = [0.9] * 30 + [0.1] * 5
    assert current_method(scores_high, score_thresh=0.5, susp_limit=17, inf_limit=27) == "INFESTED"


def test_current_method_boundary_values():
    scores = [0.9] * 17
    assert current_method(scores, score_thresh=0.5, susp_limit=17, inf_limit=27) == "SUSPICIOUS"
    scores = [0.9] * 16
    assert current_method(scores, score_thresh=0.5, susp_limit=17, inf_limit=27) == "HEALTHY"
    scores = [0.9] * 27
    assert current_method(scores, score_thresh=0.5, susp_limit=17, inf_limit=27) == "SUSPICIOUS"
    scores = [0.9] * 28
    assert current_method(scores, score_thresh=0.5, susp_limit=17, inf_limit=27) == "INFESTED"

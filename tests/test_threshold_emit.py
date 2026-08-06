"""evaluate_decision_rules.py must be able to emit the fitted cutoff as a
decision_threshold.json the app can load.

Without this the fitted value only ever appeared as a 3-decimal number inside a
candidate's display name, so shipping a refit meant retyping a rounded number by hand.
"""
import json

import evaluate_decision_rules as ev
from app.decision.threshold import DEFAULT_SPAN, ThresholdConfig


def _records(n=40):
    """Separable synthetic sessions: F sessions score low, T sessions score high."""
    out = []
    for i in range(n):
        infested = i % 2 == 0
        base = 0.8 if infested else 0.2
        scores = [base + 0.01 * (j % 5) for j in range(36)]
        out.append(
            ev.SessionRecord(
                path=f"s{i}.wav",
                label="T" if infested else "F",
                month_bucket="jan",
                scores=scores,
            )
        )
    return out


def test_ewma_candidate_exposes_its_fitted_cutoff_at_full_precision():
    candidates = ev.build_candidates(_records())
    ewma = next(c for c in candidates if c["name"].startswith("ewma_peak(") and c["span"] == DEFAULT_SPAN)

    assert "fitted_cutoff" in ewma
    assert isinstance(ewma["fitted_cutoff"], float)
    assert ewma["span"] == DEFAULT_SPAN


def test_ewma_span_sweep_produces_one_candidate_per_span():
    candidates = ev.build_candidates(_records())
    ewma_spans = {c["span"] for c in candidates if c["name"].startswith("ewma_peak(")}

    assert ewma_spans == set(ev.EWMA_SPAN_SWEEP)


def test_write_threshold_config_round_trips_through_the_apps_loader(tmp_path):
    out = tmp_path / "decision_threshold.json"
    cutoff = 0.5696557745704712  # full float precision must survive the write

    ev.write_threshold_config(str(out), cutoff, DEFAULT_SPAN)

    config = ThresholdConfig.from_json(out.read_text())
    assert config.cutoff == cutoff
    assert config.span == DEFAULT_SPAN


def test_written_config_declares_the_ewma_peak_method(tmp_path):
    out = tmp_path / "decision_threshold.json"
    ev.write_threshold_config(str(out), 0.5, DEFAULT_SPAN)

    assert json.loads(out.read_text())["method"] == "ewma_peak"


def test_quantile_cross_check_is_reported_for_the_ewma_statistic():
    # The original fit was validated by two independent methods agreeing to within
    # 0.0004; that check is only possible if the quantile band is surfaced too.
    candidates = ev.build_candidates(_records())
    band = ev.ewma_quantile_cross_check(candidates, DEFAULT_SPAN)

    assert band is not None
    assert band.t_low <= band.t_high


def test_quantile_cross_check_is_span_specific():
    # A band fitted at one span must not be reported against a cutoff fitted at another --
    # that would silently compare two different statistics.
    candidates = ev.build_candidates(_records())
    other_span = next(s for s in ev.EWMA_SPAN_SWEEP if s != DEFAULT_SPAN)

    band_default = ev.ewma_quantile_cross_check(candidates, DEFAULT_SPAN)
    band_other = ev.ewma_quantile_cross_check(candidates, other_span)

    assert band_default is not None and band_other is not None
    # Not asserting they differ (they could coincide by chance on tiny synthetic data) --
    # just that each call is actually filtering by its own span, not silently mixing.
    default_bands = {
        c["quantile_thresholds"]
        for c in candidates
        if c["name"].startswith("fixed_n_quantile_ewma_peak_span") and c.get("span") == DEFAULT_SPAN
    }
    assert band_default in default_bands

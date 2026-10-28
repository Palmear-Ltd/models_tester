import json
import math

import numpy as np
import pytest

from app.decision.noisy_test import (
    CONTINUE,
    DONE,
    EXTENDING,
    REPEAT,
    HEALTHY,
    INFESTED,
    TOO_NOISY,
    NoisyTestConfig,
    NoisyTestSession,
    default_config,
    hop_floor_db,
)
from app.decision.threshold import EwmaPeakDecision, ThresholdConfig

SR = 44100
HOP = SR // 2
CUTOFF = 0.5696557745704712


def ewma_verdict(scores, cutoff=CUTOFF, span=5.0):
    acc = EwmaPeakDecision(ThresholdConfig(cutoff=cutoff, span=span))
    for s in scores:
        acc.update(s)
    return acc.state, acc.peak


def on(**kw):
    base = dict(enabled=True, noise_db=6.0, cutoff=CUTOFF, span=5.0)
    base.update(kw)
    return NoisyTestConfig(**base)


def run(session, floors, scores):
    """Feed per-hop floors; scores[w] is the score of the window ending at hop w + 4.
    Returns the list of actions."""
    actions = []
    for h, f in enumerate(floors):
        w = h - (session.config.hops_per_window - 1)
        score = scores[w] if 0 <= w < len(scores) else None
        actions.append(session.feed_floor(f, score))
        if actions[-1].kind in (DONE, REPEAT):
            break
    return actions


# --- hop_floor_db -----------------------------------------------------------------------

def test_floor_ignores_sparse_clicks_but_follows_broadband_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 10 ** (-50 / 20), HOP)
    clicky = base.copy()
    idx = rng.choice(HOP, 15, replace=False)
    clicky[idx] += 0.05
    loud = base * 10 ** (12 / 20)
    f0 = hop_floor_db(base)
    assert abs(hop_floor_db(clicky) - f0) < 1.0
    assert hop_floor_db(loud) - f0 > 11.0


def test_floor_of_silence_and_garbage_is_minus_inf():
    assert hop_floor_db(np.zeros(HOP)) == -math.inf
    assert hop_floor_db(np.array([])) == -math.inf
    assert hop_floor_db(np.array([np.nan] * 10)) == -math.inf


# --- config -----------------------------------------------------------------------------

def test_shipped_default_is_off_and_never_flags():
    c = NoisyTestConfig()
    assert c.enabled is False
    assert math.isinf(c.noise_db) and math.isinf(c.abs_noise_db)


def test_config_json_round_trip(tmp_path):
    c = on(max_extra_sec=40.0, max_attempts=3, abs_noise_db=-30.0)
    p = tmp_path / "n.json"
    p.write_text(c.to_json())
    assert default_config(str(p)) == c
    assert default_config(str(tmp_path / "missing.json")) == NoisyTestConfig()


def test_infinite_thresholds_survive_json():
    c = NoisyTestConfig()
    assert NoisyTestConfig.from_json(c.to_json()) == c
    json.loads(c.to_json())  # strict JSON, no Infinity literal


# --- disabled = today's behaviour -------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_disabled_is_todays_ewma_verdict_after_40_hops(seed):
    rng = np.random.default_rng(seed)
    scores = list(rng.uniform(0, 1, 36))
    floors = list(rng.uniform(-70, -20, 60))  # wild noise must not matter when off
    s = NoisyTestSession(NoisyTestConfig(enabled=False, noise_db=1.0, cutoff=CUTOFF))
    acts = run(s, floors, scores)
    assert len(acts) == 40 and acts[-1].kind == DONE
    assert all(a.kind == CONTINUE for a in acts[:-1])
    state, peak = ewma_verdict(scores)
    r = acts[-1].result
    assert r.verdict == state and r.provisional_verdict == state
    assert r.quiet_peak == pytest.approx(peak) and r.attempt == 1


def test_finish_early_when_disabled_uses_windows_so_far():
    scores = [0.9] * 10
    s = NoisyTestSession(NoisyTestConfig(enabled=False, cutoff=CUTOFF))
    run(s, [-50.0] * 14, scores)
    assert s.finish().verdict == INFESTED


# --- enabled ----------------------------------------------------------------------------

def test_all_quiet_is_identical_to_today():
    rng = np.random.default_rng(7)
    scores = list(rng.uniform(0, 1, 36))
    s = NoisyTestSession(on(max_extra_sec=40.0, max_attempts=2))
    acts = run(s, [-55.0] * 40, scores)
    assert len(acts) == 40 and acts[-1].kind == DONE
    assert acts[-1].result.verdict == ewma_verdict(scores)[0]
    assert acts[-1].result.n_quiet == 36


def test_burst_extends_then_decides_on_first_36_quiet_windows():
    floors = [-55.0] * 120
    floors[10:16] = [-35.0] * 6           # a car: 6 loud hops
    scores = [0.1] * 116
    for w in range(6, 16):                # windows touching hops 10..15 score high
        scores[w] = 0.95
    s = NoisyTestSession(on(max_extra_sec=40.0))
    acts = run(s, floors, scores)
    kinds = [a.kind for a in acts]
    assert kinds.count(EXTENDING) == 1 and kinds.index(EXTENDING) == 39
    r = acts[-1].result
    assert acts[-1].kind == DONE and r.verdict == HEALTHY   # the noisy high scores never count
    assert r.n_quiet == 36 and sum(r.quiet_mask) >= 36
    assert r.provisional_verdict == INFESTED                # all windows: today's tester answer
    assert len(acts) == 40 + 10                             # 10 noisy windows replaced


def test_noisy_windows_never_count_towards_infested_either():
    floors = [-55.0] * 40
    floors[20:25] = [-30.0] * 5
    scores = [0.05] * 36
    for w in range(16, 25):
        scores[w] = 0.99
    s = NoisyTestSession(on(max_extra_sec=0.0, max_attempts=1))
    r = run(s, floors, scores)[-1].result
    assert r.verdict == TOO_NOISY and r.provisional_verdict == INFESTED


def test_warn_only_gives_too_noisy_at_20s():
    floors = [-55.0] * 40
    floors[30] = -20.0
    s = NoisyTestSession(on())
    acts = run(s, floors, [0.1] * 36)
    assert len(acts) == 40 and acts[-1].kind == DONE and acts[-1].result.verdict == TOO_NOISY


def test_extension_exhausted_then_repeat_then_too_noisy():
    loud = [-55.0, -30.0] * 200          # every other hop noisy: no window is ever quiet
    s = NoisyTestSession(on(max_extra_sec=10.0, max_attempts=2))
    a1 = run(s, loud, [0.1] * 400)
    assert a1[-1].kind == REPEAT and len(a1) == 60
    assert a1[-1].result.verdict == TOO_NOISY and a1[-1].result.attempt == 1
    s.start_attempt()
    a2 = run(s, loud, [0.1] * 400)
    assert a2[-1].kind == DONE and a2[-1].result.verdict == TOO_NOISY and a2[-1].result.attempt == 2


def test_repeat_attempt_starts_from_zero_and_can_succeed():
    s = NoisyTestSession(on(max_attempts=2))
    noisy = [-55.0] * 40
    noisy[20] = -25.0
    assert run(s, noisy, [0.1] * 36)[-1].kind == REPEAT
    s.start_attempt()
    acts = run(s, [-55.0] * 40, [0.9] * 36)
    assert acts[-1].kind == DONE and acts[-1].result.verdict == INFESTED and acts[-1].result.attempt == 2


def test_reference_is_judged_on_the_whole_attempt_not_just_the_start():
    # The test starts during a car: the first hops are loud. Judged later against the quiet
    # part, those early windows are noisy.
    floors = [-30.0] * 8 + [-55.0] * 100
    s = NoisyTestSession(on(max_extra_sec=40.0))
    acts = run(s, floors, [0.1] * 104)
    r = acts[-1].result
    assert r.verdict == HEALTHY and not any(r.quiet_mask[:8])


def test_absolute_threshold_catches_noise_that_never_stops():
    s = NoisyTestSession(on(noise_db=math.inf, abs_noise_db=-40.0))
    r = run(s, [-35.0] * 40, [0.1] * 36)[-1].result
    assert r.verdict == TOO_NOISY


def test_silent_hops_are_not_noise_and_do_not_break_the_reference():
    floors = [-math.inf] * 3 + [-55.0] * 37
    s = NoisyTestSession(on())
    r = run(s, floors, [0.2] * 36)[-1].result
    assert r.verdict == HEALTHY and r.n_quiet == 36


def test_finish_early_when_enabled_and_short_of_quiet_windows_is_too_noisy():
    s = NoisyTestSession(on(max_extra_sec=40.0))
    run(s, [-55.0] * 20, [0.9] * 16)
    assert s.finish().verdict == TOO_NOISY


def test_result_records_what_was_seen():
    s = NoisyTestSession(on())
    r = run(s, [-55.0] * 40, [0.3] * 36)[-1].result
    assert r.n_hops == 40 and r.n_windows == 36 and len(r.floors_db) == 40
    assert len(r.quiet_mask) == 36 and r.reference_db == pytest.approx(-55.0)
    d = r.to_dict()
    json.dumps(d)
    assert d["verdict"] == HEALTHY

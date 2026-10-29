"""Generates tests/data/noisy_test_vectors.json: golden vectors for porting
app/decision/noisy_test.py to the phone (Dart).

Two kinds of vector:
  * "floor": a short hop of samples and the expected hop_floor_db (checks the trimmed-floor
    port, including NumPy's default linear-interpolation percentile).
  * "session": a config and, per attempt, the per-hop floors (dBFS, null = -inf) and window
    scores (null while the first window fills), with the expected Action kind for every hop
    and the expected AttemptResult when the attempt closes.

The port passes if it reproduces every action and every result field. Float fields compare
to 1e-9. Run from the repo root:  ../.venv/bin/python scripts/make_noisy_test_vectors.py
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.decision.noisy_test import DONE, REPEAT, NoisyTestConfig, NoisyTestSession, hop_floor_db  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data",
                   "noisy_test_vectors.json")
FITTED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "9_1_2",
                      "noisy_test.json")
CUTOFF = 0.5696557745704712
K = 5  # hops per window


def enc(v):
    return None if isinstance(v, float) and math.isinf(v) else v


def cfg_dict(c):
    d = json.loads(c.to_json())
    d.pop("method")
    return d


def run_attempt(session, floors, scores):
    """Feeds one attempt. scores[w] is the window ending at hop w + K - 1. Stops at DONE/REPEAT."""
    hops, actions, result = [], [], None
    for h, f in enumerate(floors):
        w = h - (K - 1)
        s = scores[w] if 0 <= w < len(scores) else None
        a = session.feed_floor(f, s)
        hops.append([enc(float(f)), None if s is None else float(s)])
        actions.append(a.kind)
        if a.kind in (DONE, REPEAT):
            result = a.result.to_dict()
            break
    if result is None:
        raise RuntimeError("attempt did not close; give it more hops")
    return {"hops": hops, "actions": actions, "result": result}


def session_vector(name, note, cfg, attempts):
    s = NoisyTestSession(cfg)
    out = []
    for i, (floors, scores) in enumerate(attempts):
        if i:
            s.start_attempt()
        out.append(run_attempt(s, floors, scores))
    if out[-1]["actions"][-1] != DONE:
        raise RuntimeError(f"{name}: last attempt must end DONE")
    return {"kind": "session", "name": name, "note": note, "config": cfg_dict(cfg), "attempts": out}


def on(**kw):
    base = dict(enabled=True, noise_db=10.0, cutoff=CUTOFF, span=5.0)
    base.update(kw)
    return NoisyTestConfig(**base)


def main():
    rng = np.random.default_rng(20261029)
    r = lambda n, lo=0.0, hi=1.0: [round(float(v), 6) for v in rng.uniform(lo, hi, n)]  # noqa: E731
    vectors = []

    # --- floor vectors ------------------------------------------------------------------
    for name, x in [
        ("floor_noise", rng.normal(0, 0.003, 400)),
        ("floor_noise_with_clicks", np.where(np.arange(400) % 97 == 0, 0.2, rng.normal(0, 0.003, 400))),
        ("floor_loud", rng.normal(0, 0.05, 400)),
        ("floor_silence", np.zeros(400)),
    ]:
        x = [round(float(v), 6) for v in x]
        vectors.append({"kind": "floor", "name": name, "trim_percentile": 90.0, "samples": x,
                        "expected_db": enc(hop_floor_db(np.array(x), 90.0))})

    # --- session vectors ----------------------------------------------------------------
    quiet = lambda n, lvl=-55.0: [round(lvl + float(v), 3) for v in rng.uniform(-2, 2, n)]  # noqa: E731
    fitted = NoisyTestConfig.from_json(open(FITTED).read())

    vectors.append(session_vector(
        "shipped_off_quiet", "Shipped models/9_1_2/noisy_test.json (enabled=false): today's 20 s verdict.",
        fitted, [(quiet(40), r(36))]))
    loud = quiet(40)
    loud[10:20] = [-25.0] * 10
    vectors.append(session_vector(
        "shipped_off_noise_ignored", "Off: noise changes nothing; DONE at hop 40.", fitted,
        [(loud, r(36, 0.6, 1.0))]))

    on_fitted = NoisyTestConfig.from_json(json.dumps(dict(cfg_dict(fitted), enabled=True)))
    vectors.append(session_vector(
        "fitted_on_all_quiet_healthy", "Fitted config switched on, no noise: same as today, HEALTHY.",
        on_fitted, [(quiet(40), r(36, 0.0, 0.4))]))
    vectors.append(session_vector(
        "fitted_on_all_quiet_infested", "Fitted config switched on, no noise: same as today, INFESTED.",
        on_fitted, [(quiet(40), r(36, 0.5, 1.0))]))

    fl = quiet(120)
    fl[12:20] = [-38.0] * 8
    sc = r(116, 0.0, 0.3)
    for w in range(8, 20):
        sc[w] = 0.95
    vectors.append(session_vector(
        "fitted_on_burst_extends_healthy",
        "A noisy burst with high scores: EXTENDING at hop 40, the noisy windows never count -> HEALTHY "
        "(provisional INFESTED).", on_fitted, [(fl, sc)]))

    fl = quiet(120)
    fl[25:30] = [-30.0] * 5
    sc = r(116, 0.0, 0.3)
    for w in range(40, 50):
        sc[w] = 0.9
    vectors.append(session_vector(
        "fitted_on_burst_extends_infested",
        "Noise mid-test; infested evidence in the quiet windows after the extension -> INFESTED.",
        on_fitted, [(fl, sc)]))

    fl = quiet(40)
    fl[20] = -30.0
    vectors.append(session_vector(
        "warn_only_too_noisy", "max_extra_sec=0, max_attempts=1: TOO_NOISY at hop 40.",
        on(), [(fl, r(36))]))

    fl = [(-55.0 if h % 2 == 0 else -30.0) for h in range(200)]
    vectors.append(session_vector(
        "extension_exhausted_too_noisy", "Noise never stops: extension of 10 s runs out -> TOO_NOISY at hop 60.",
        on(max_extra_sec=10.0), [(fl, r(196))]))

    fl1 = quiet(40)
    fl1[20] = -30.0
    vectors.append(session_vector(
        "repeat_then_infested", "Attempt 1 noisy -> REPEAT; attempt 2 quiet -> INFESTED.",
        on(max_attempts=2), [(fl1, r(36)), (quiet(40), r(36, 0.7, 1.0))]))

    fl = [(-55.0 if h % 2 == 0 else -30.0) for h in range(80)]
    vectors.append(session_vector(
        "repeat_then_too_noisy", "Extend 5 s, 2 attempts, noise throughout -> REPEAT then TOO_NOISY.",
        on(max_extra_sec=5.0, max_attempts=2), [(fl, r(76)), (fl, r(76))]))

    fl = [-28.0] * 8 + quiet(112)
    vectors.append(session_vector(
        "noise_at_start_judged_later",
        "Test starts during a car: the first hops are judged noisy once the quiet reference exists.",
        on(max_extra_sec=40.0), [(fl, r(116, 0.0, 0.5))]))

    fl = [-math.inf] * 3 + quiet(37)
    vectors.append(session_vector(
        "silent_hops_not_noise", "Floors of -inf (digital silence) are not noise and do not move the reference.",
        on(), [(fl, r(36, 0.0, 0.4))]))

    vectors.append(session_vector(
        "absolute_threshold", "abs_noise_db=-40: constant loud noise -> TOO_NOISY.",
        on(noise_db=math.inf, abs_noise_db=-40.0), [(quiet(40, -35.0), r(36))]))

    doc = {
        "about": "Golden vectors for app/decision/noisy_test.py. Generated by scripts/make_noisy_test_vectors.py; "
                 "do not edit by hand. null floor = -inf; null score = no window yet. Floats compare to 1e-9.",
        "vectors": vectors,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {len(vectors)} vectors to {OUT}")


if __name__ == "__main__":
    main()

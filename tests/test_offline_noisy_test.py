"""offline_score.py --noisy-test: the offline replay must give the same attempts as
NoisyTestSession fed directly, and run end to end on a real WAV with the 9_1_2 model."""
import argparse
import csv
import json
import os

import numpy as np
import pytest
import soundfile as sf

offline_score = pytest.importorskip("offline_score")

from app.decision.noisy_test import DONE, REPEAT, NoisyTestConfig, NoisyTestSession  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
CUTOFF = 0.5696557745704712


def direct(cfg, floors, scores, pause_hops):
    """Reference: feed the session by hand, attempt by attempt, test after test."""
    out, h0, test = [], 0, 0
    while h0 < len(floors):
        test += 1
        s = NoisyTestSession(cfg)
        while h0 < len(floors):
            start, kind = h0, None
            for h in range(start, len(floors)):
                a = s.feed_floor(floors[h], scores[h - 4] if h - start >= 4 else None)
                if a.kind in (DONE, REPEAT):
                    kind = a.kind
                    break
            r = a.result if kind else s.finish()
            out.append((test, r.attempt, start, r.verdict, r.n_hops, kind is not None))
            if kind is None:
                return out
            h0 = h + 1
            if kind == DONE:
                break
            h0 += pause_hops
            s.start_attempt()
    return out


@pytest.mark.parametrize("extra,attempts", [(0.0, 1), (40.0, 1), (0.0, 2), (20.0, 2)])
def test_replay_matches_the_session_fed_directly(extra, attempts):
    rng = np.random.default_rng(3)
    floors = list(-55 + rng.uniform(-2, 2, 600))
    for a in range(30, 600, 70):                     # a car every 35 s, 4 s long
        floors[a:a + 8] = [-30.0] * 8
    scores = list(rng.uniform(0, 1, 596))
    cfg = NoisyTestConfig(enabled=True, noise_db=10.0, max_extra_sec=extra, max_attempts=attempts,
                          cutoff=CUTOFF, span=5.0)
    rows = offline_score.noisy_replay(cfg, floors, scores)
    got = [(r["test"], r["attempt"], int(r["start_sec"] * 2), r["verdict"], int(r["duration_sec"] * 2), r["complete"])
           for r in rows]
    assert got == direct(cfg, floors, scores, pause_hops=6)
    assert len(rows) > 5


def test_disabled_config_is_back_to_back_20s_tests():
    floors = [-55.0] * 200
    scores = [0.1] * 196
    rows = offline_score.noisy_replay(NoisyTestConfig(enabled=False, cutoff=CUTOFF), floors, scores)
    assert [r["duration_sec"] for r in rows] == [20.0] * 5
    assert all(r["complete"] for r in rows) and all(r["verdict"] == "HEALTHY" for r in rows)


def test_end_to_end_on_a_wav(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10 ** (-55 / 20), 50 * SR).astype(np.float32)
    x[5 * SR:8 * SR] *= 10 ** (30 / 20)
    d = tmp_path / "corpus" / "F"
    d.mkdir(parents=True)
    sf.write(d / "TN_9_1_2_20261029_101010.wav", x, SR)
    args = argparse.Namespace(
        corpus=[str(tmp_path / "corpus")], model=os.path.join(REPO, "models", "9_1_2", "model.tflite"),
        scaler=os.path.join(REPO, "models", "9_1_2", "scaler.json"), cache_dir=str(tmp_path / "cache"),
        manifest_out=str(tmp_path / "m.csv"), limit=None, glob=None, workers=1, max_duration_sec=20.0,
        noisy_test=os.path.join(REPO, "models", "9_1_2", "noisy_test.json"), noisy_out=str(tmp_path / "n.csv"),
        noisy_max_extra_sec=None, noisy_max_attempts=None,
    )
    rows = offline_score.run(args)
    with open(tmp_path / "n.csv") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    first = rows[0]
    assert first["test"] == 1 and first["complete"] and first["duration_sec"] > 20.0   # the burst forced an extension
    assert first["n_quiet"] == 36
    # The scores came from the uncapped cache: one per hop after the first 4.
    cache = json.load(open(next((tmp_path / "cache").iterdir())))
    assert len(cache["scores"]) == 100 - 4

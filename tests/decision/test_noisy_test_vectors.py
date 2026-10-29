"""Replays tests/data/noisy_test_vectors.json (the golden vectors handed to the phone port)
through app/decision/noisy_test.py. Regenerate with scripts/make_noisy_test_vectors.py."""
import json
import math
import os

import numpy as np
import pytest

from app.decision.noisy_test import NoisyTestConfig, NoisyTestSession, hop_floor_db

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "noisy_test_vectors.json")
with open(PATH) as f:
    VECTORS = json.load(f)["vectors"]


def dec(v):
    return -math.inf if v is None else v


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("v", [v for v in VECTORS if v["kind"] == "floor"], ids=lambda v: v["name"])
def test_floor_vector(v):
    got = hop_floor_db(np.array(v["samples"]), v["trim_percentile"])
    want = dec(v["expected_db"])
    assert got == want if math.isinf(want) else got == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("v", [v for v in VECTORS if v["kind"] == "session"], ids=lambda v: v["name"])
def test_session_vector(v):
    s = NoisyTestSession(NoisyTestConfig.from_json(json.dumps(v["config"])))
    for i, att in enumerate(v["attempts"]):
        if i:
            s.start_attempt()
        kinds, result = [], None
        for floor, score in att["hops"]:
            a = s.feed_floor(dec(floor), score)
            kinds.append(a.kind)
            result = a.result
        assert kinds == att["actions"]
        got = result.to_dict()
        assert set(got) == set(att["result"])
        for k, want in att["result"].items():
            assert same(got[k], want), k


def test_vectors_cover_every_outcome():
    seen = set()
    for v in VECTORS:
        for att in v.get("attempts", []):
            seen.update(att["actions"])
            seen.add(att["result"]["verdict"])
    assert {"CONTINUE", "EXTENDING", "REPEAT", "DONE", "HEALTHY", "INFESTED", "TOO_NOISY"} <= seen

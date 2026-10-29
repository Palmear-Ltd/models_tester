"""Extend/repeat simulation on real 2026 recordings, with the goal of reducing false alarms.

Virtual tracks: 5 consecutive 20 s recordings of the same session and label, chained at the
level of cached per-hop floors and 9_1_2 window scores (no audio). Recording k >= 2 drops
its first 4 hops so every chained hop carries a real window score. So the first 4 windows
after a join see 4 floors from the previous recording: an approximation, noted in the spec.

Usage: sim_extend.py TABLE_CSV OUT_CSV
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from app.decision.noisy_test import HEALTHY, INFESTED, TOO_NOISY, NoisyTestConfig
from noisy_test_injection import replay

TABLE, OUT = sys.argv[1], sys.argv[2]
th = json.load(open("models/9_1_2/decision_threshold.json"))
CUT, SPAN = th["cutoff"], th.get("span", 5.0)
JOIN = 5

d = pd.read_csv(TABLE)
d = d[d.role != "manual_test"]
d["split"] = np.where(d.role.isin(["train_main", "val"]), "choose", "confirm")
d = d.assign(ts=d.basename.str.extract(r"(\d{8}_\d{6})")[0]).sort_values(["session_id", "ts"])
tracks = []
for (sess, lab, split), g in d.groupby(["session_id", "label", "split"]):
    rows = list(g.itertuples())
    for i in range(0, len(rows) - JOIN + 1, JOIN):
        fl, sc = [], []
        for k, r in enumerate(rows[i:i + JOIN]):
            f = json.loads(r.floors); s = json.loads(r.scores)
            if len(f) != 40 or len(s) != 36:
                break
            fl += f if k == 0 else f[4:]
            sc += s
        else:
            tracks.append(dict(session=sess, label=lab, split=split, floors=fl, scores=sc))
print(pd.Series([(t["split"], t["label"]) for t in tracks]).value_counts().to_dict(), flush=True)

MODES = {
    "warn": dict(max_extra_sec=0.0, max_attempts=1),
    "extend20": dict(max_extra_sec=20.0, max_attempts=1),
    "extend40": dict(max_extra_sec=40.0, max_attempts=1),
    "repeat2": dict(max_extra_sec=0.0, max_attempts=2),
    "extend20_repeat2": dict(max_extra_sec=20.0, max_attempts=2),
}
DETECTORS = [dict(noise_db=x, abs_noise_db=float("inf")) for x in (6.0, 8.0, 10.0, 12.0, 15.0)]
DETECTORS += [dict(noise_db=float("inf"), abs_noise_db=a) for a in (-45.0, -40.0, -35.0)]
DETECTORS += [dict(noise_db=10.0, abs_noise_db=-40.0)]

rng = np.random.default_rng(1)


def summarise(verdicts):
    v = pd.Series(verdicts)
    return {k: float((v == k).mean()) for k in (HEALTHY, INFESTED, TOO_NOISY, "RAN_OUT")}


off = NoisyTestConfig(enabled=False, cutoff=CUT, span=SPAN)
rows = []
for split in ("choose", "confirm"):
    for lab in ("F", "T"):
        ts = [t for t in tracks if t["split"] == split and t["label"] == lab]
        today = summarise([replay(off, t["floors"], t["scores"]).verdict for t in ts])
        rows.append(dict(split=split, label=lab, detector="today", mode="-", null=False, n=len(ts), **today))
for det in DETECTORS:
    for mname, m in MODES.items():
        cfg = NoisyTestConfig(enabled=True, cutoff=CUT, span=SPAN, **det, **m)
        for split in ("choose", "confirm"):
            for lab in ("F", "T"):
                ts = [t for t in tracks if t["split"] == split and t["label"] == lab]
                real, null = [], []
                for t in ts:
                    r = replay(cfg, t["floors"], t["scores"])
                    real.append(r.verdict if r else "RAN_OUT")
                    # Null: flag the same share of hops at random (forced 200 dB loud).
                    probe = replay(NoisyTestConfig(enabled=True, cutoff=CUT, span=SPAN, **det, max_extra_sec=1e4),
                                   t["floors"], t["scores"])
                    rate = 1 - (np.mean(probe.quiet_mask) if probe and probe.quiet_mask else 1.0)
                    hop_rate = 1 - (1 - rate) ** (1 / 5) if rate < 1 else 1.0
                    fl = [200.0 if rng.random() < hop_rate else f for f in t["floors"]]
                    rn = replay(NoisyTestConfig(enabled=True, cutoff=CUT, span=SPAN, noise_db=50.0, **m), fl, t["scores"])
                    null.append(rn.verdict if rn else "RAN_OUT")
                name = f"rel{det['noise_db']}|abs{det['abs_noise_db']}"
                rows.append(dict(split=split, label=lab, detector=name, mode=mname, null=False, n=len(ts), **summarise(real)))
                rows.append(dict(split=split, label=lab, detector=name, mode=mname, null=True, n=len(ts), **summarise(null)))
        print(det, mname, flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)

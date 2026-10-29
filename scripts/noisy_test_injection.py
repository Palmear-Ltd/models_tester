#!/usr/bin/env python
"""Injection evidence for noisy test handling
(docs/superpowers/specs/2026-10-28-noisy-test-handling-design.md, "Fitting the threshold").

`build` makes long test tracks and caches, per track and per noise variant, the per-hop
measurements and the 9_1_2 score of every window:

  * primary tracks: 4 consecutive 20 s recordings of the same 2026 session and label, joined
    with a 50 ms crossfade (80 s). 9_1_2 (committed 2025-12-29) has never heard 2026 audio.
    Joined recordings are a stand-in for one long recording of one tree.
  * secondary tracks: real long recordings from 2025-12 (60 / 120 s), cut to at most 80 s.
    They are continuous, but 9_1_2 may have trained on them.
  * variants: clean, plus two random stretches of the long roadside car recording mixed in at
    its recorded level.

`eval` (added with the portable module) replays the cached tracks through the noisy test
state machine. WAV I/O and TFLite live here, outside app/, like calibrate.py.

  ../.venv/bin/python scripts/noisy_test_injection.py build --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import offline_score as o  # noqa: E402

WORKSPACE = os.path.dirname(REPO)
DATASET = os.path.join(WORKSPACE, "training/one_shot_1/dataset.csv")
CATALOG = os.path.join(WORKSPACE, "audio/catalog.csv")
AUDIO_ROOT = os.path.join(WORKSPACE, "audio")
CAR_WAV = os.path.join(WORKSPACE, "long_recording_for_cars/2026-10-21 10.29.32.wav")
CACHE = os.path.join(WORKSPACE, "training/9_1_5/.noisy_test_cache")
MODEL = os.path.join(REPO, "models/9_1_2")

SR = o.TARGET_SR
HOP = int(SR * o.HOP_SEC)
WIN = int(SR * o.WINDOW_SEC)
TRACK_SEC = 80
JOIN = 4
FADE = int(0.05 * SR)
LF_EDGE_HZ = 300.0
TRIM_PERCENTILES = (90.0, 95.0)


def hop_measures(hop):
    """Trimmed floors (dB) at each TRIM_PERCENTILES, full RMS (dB) and the share of power
    below LF_EDGE_HZ, for one 0.5 s hop."""
    x = np.asarray(hop, dtype=np.float64)
    full = float(np.sqrt(np.mean(x * x)))
    out = {"full_db": 20 * np.log10(full + 1e-12)}
    for p in TRIM_PERCENTILES:
        lim = float(np.percentile(np.abs(x), p))
        c = np.clip(x, -lim, lim) if lim > 0 else x
        out[f"floor{int(p)}_db"] = 20 * np.log10(float(np.sqrt(np.mean(c * c))) + 1e-12)
    xd = x - x.mean()
    pw = np.abs(np.fft.rfft(xd * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / SR)
    tot = float(pw.sum())
    out["lf_share"] = float(pw[f < LF_EDGE_HZ].sum()) / tot if tot > 0 else 0.0
    return out


def load(path, max_sec=None):
    x = o._load_wav_mono(path)
    if max_sec is not None:
        x = x[: int(max_sec * SR)]
    return x


def join(parts):
    out = parts[0].copy()
    ramp = np.linspace(0.0, 1.0, FADE, dtype=np.float32)
    for p in parts[1:]:
        tail = out[-FADE:] * (1 - ramp) + p[:FADE] * ramp
        out = np.concatenate([out[:-FADE], tail, p[FADE:]])
    return out


def build_tracks(rng):
    d = pd.read_csv(DATASET)
    d = d[(d.year == 2026) & d.role.isin(["val", "recent", "test"]) & (d.duration_s.round(1) == 20.0)]
    d = d.assign(ts=d.basename.str.extract(r"(\d{8}_\d{6})")[0]).sort_values(["session_id", "ts"])
    tracks = []
    for (sess, lab), g in d.groupby(["session_id", "label"]):
        paths = g.abs_path.tolist()
        for i in range(0, len(paths) - JOIN + 1, JOIN):
            tracks.append({"kind": "joined", "label": lab, "session": sess, "members": paths[i:i + JOIN]})
    T = [t for t in tracks if t["label"] == "T"]
    F = [t for t in tracks if t["label"] == "F"]
    F = [F[i] for i in rng.choice(len(F), size=min(len(F), 90), replace=False)]

    c = pd.read_csv(CATALOG)
    c = c[(c.year == 2025) & (c.duration_s >= 59) & c.label.isin(["T", "F"]) & c.exclude_reason.isna()
          & c.session_id.str.startswith("2025-12")]
    long_tracks = [{"kind": "long", "label": r.label, "session": r.session_id,
                    "members": [os.path.join(AUDIO_ROOT, r.path)]} for r in c.itertuples()]
    return T + F + long_tracks


def window_scores(audio):
    m = o._worker_model
    return [o._infer_score(audio[k * HOP - WIN:k * HOP], m, o._worker_scaler, o._worker_feature_extractor,
                           o._worker_seq_len, o._worker_n_mels, o._worker_prep_params)
            for k in range(WIN // HOP, len(audio) // HOP + 1)]


def track_job(args):
    idx, track, noise_starts = args
    out_path = os.path.join(CACHE, f"track_{idx:04d}.json")
    if os.path.exists(out_path):
        return out_path
    if track["kind"] == "joined":
        audio = join([load(p, 20) for p in track["members"]])
    else:
        audio = load(track["members"][0], TRACK_SEC)
    n = (len(audio) // HOP) * HOP
    audio = audio[:n]
    car = np.load(os.path.join(CACHE, "car.npy"), mmap_mode="r")
    variants = {"clean": audio}
    for j, s in enumerate(noise_starts):
        variants[f"car{j}"] = np.clip(audio + np.asarray(car[s:s + n], dtype=np.float32), -1, 1)
    rec = dict(track, idx=idx, n_hops=n // HOP, noise_starts=noise_starts, variants={})
    for name, x in variants.items():
        hops = [hop_measures(x[h * HOP:(h + 1) * HOP]) for h in range(n // HOP)]
        rec["variants"][name] = {
            "hops": {k: [h[k] for h in hops] for k in hops[0]},
            "scores": window_scores(x),
        }
    with open(out_path, "w") as f:
        json.dump(rec, f)
    return out_path


def build(args):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(20261028)
    car_path = os.path.join(CACHE, "car.npy")
    if not os.path.exists(car_path):
        np.save(car_path, o._load_wav_mono(CAR_WAV))
    car_len = len(np.load(car_path, mmap_mode="r"))
    tracks = build_tracks(rng)
    jobs = []
    for i, t in enumerate(tracks):
        span = TRACK_SEC * SR
        jobs.append((i, t, [int(rng.integers(0, car_len - span)) for _ in range(2)]))
    with open(os.path.join(CACHE, "index.json"), "w") as f:
        json.dump([{"idx": i, "kind": t["kind"], "label": t["label"], "session": t["session"]} for i, t, _ in jobs], f)
    print(f"{len(jobs)} tracks:", pd.Series([f"{t['kind']}/{t['label']}" for t in tracks]).value_counts().to_dict())
    init = (os.path.join(MODEL, "model.tflite"), os.path.join(MODEL, "scaler.json"), dict(o.DEFAULT_PREP_PARAMS))
    with ProcessPoolExecutor(args.workers, initializer=o._init_worker, initargs=init) as ex:
        for k, _ in enumerate(ex.map(track_job, jobs, chunksize=1), 1):
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)


CONFIGS = {
    "warn": dict(max_extra_sec=0.0, max_attempts=1),
    "extend40": dict(max_extra_sec=40.0, max_attempts=1),
    "repeat2": dict(max_extra_sec=0.0, max_attempts=2),
    "extend20_repeat2": dict(max_extra_sec=20.0, max_attempts=2),
}


def replay(cfg, floors, scores, start_hop=0):
    """Feed a cached track through NoisyTestSession from `start_hop`, continuing through
    the track for repeats after a `repeat_pause_sec` gap, like a live road. Returns the
    final AttemptResult, or None if the track ran out first."""
    from app.decision.noisy_test import DONE, REPEAT, NoisyTestSession
    s = NoisyTestSession(cfg)
    k = cfg.hops_per_window - 1
    h0 = start_hop
    pause = int(round(cfg.repeat_pause_sec / 0.5))
    while True:
        for i in range(len(floors) - h0):
            h = h0 + i
            score = scores[h - k] if i >= k else None   # scores[j] = window ending at hop j + k
            a = s.feed_floor(floors[h], score)
            if a.kind == DONE:
                return a.result
            if a.kind == REPEAT:
                break
        else:
            return None
        h0 = h + 1 + pause
        s.start_attempt()


def load_cache():
    with open(os.path.join(CACHE, "index.json")) as f:
        index = json.load(f)
    tracks = []
    for row in index:
        with open(os.path.join(CACHE, f"track_{row['idx']:04d}.json")) as f:
            tracks.append(json.load(f))
    return tracks


def evaluate(args):
    from app.decision.noisy_test import HEALTHY, INFESTED, TOO_NOISY, NoisyTestConfig
    from app.decision.threshold import default_config as threshold_config
    th = threshold_config(os.path.join(MODEL, "decision_threshold.json"))
    tracks = load_cache()
    rows = []
    off = NoisyTestConfig(enabled=False, cutoff=th.cutoff, span=th.span)
    for trim in (90, 95):
        for q in (0.1, 0.2, 0.3):
            for ndb in args.noise_db:
                for adb in args.abs_noise_db:
                    base = dict(enabled=True, trim_percentile=float(trim), reference_quantile=q, noise_db=ndb,
                                abs_noise_db=adb, cutoff=th.cutoff, span=th.span)
                    # Bars 1 and 2: single clean 20 s recordings, warn mode.
                    warn = NoisyTestConfig(**base, **CONFIGS["warn"])
                    trig = {"T": [], "F": []}
                    for t in tracks:
                        v = t["variants"]["clean"]
                        fl, sc = v["hops"][f"floor{trim}_db"], v["scores"]
                        starts = range(0, len(fl) - 39, 40) if t["kind"] == "joined" else [0]
                        for st in starts:
                            r = replay(warn, fl[st:st + 40], sc[st:st + 36])
                            trig[t["label"]].append(r.verdict == TOO_NOISY)
                    row = dict(trim=trim, q=q, noise_db=ndb, abs_noise_db=adb,
                               trig_F=np.mean(trig["F"]), trig_T=np.mean(trig["T"]))
                    # Bar 3: whole tracks, each config, clean and car variants.
                    for cname, cextra in CONFIGS.items():
                        cfg = NoisyTestConfig(**base, **cextra)
                        for lab in ("T", "F"):
                            for var in ("clean", "car"):
                                out = []
                                for t in tracks:
                                    if t["label"] != lab or (args.kind and t["kind"] != args.kind):
                                        continue
                                    names = ["clean"] if var == "clean" else ["car0", "car1"]
                                    for nm in names:
                                        v = t["variants"][nm]
                                        r = replay(cfg, v["hops"][f"floor{trim}_db"], v["scores"])
                                        out.append(r.verdict if r else "RAN_OUT")
                                out = pd.Series(out)
                                for verdict in (HEALTHY, INFESTED, TOO_NOISY, "RAN_OUT"):
                                    row[f"{cname}|{lab}|{var}|{verdict}"] = float((out == verdict).mean())
                    rows.append(row)
                    print(f"trim{trim} q{q} ndb{ndb} adb{adb}: trigF {row['trig_F']:.3f} trigT {row['trig_T']:.3f} "
                          f"extend40 T/car HEALTHY {row['extend40|T|car|HEALTHY']:.3f} "
                          f"TOO_NOISY {row['extend40|T|car|TOO_NOISY']:.3f}", flush=True)
    # Today's behaviour (feature off) for reference.
    today = {}
    for lab in ("T", "F"):
        for var in ("clean", "car"):
            out = []
            for t in tracks:
                if t["label"] != lab or (args.kind and t["kind"] != args.kind):
                    continue
                for nm in (["clean"] if var == "clean" else ["car0", "car1"]):
                    v = t["variants"][nm]
                    out.append(replay(off, v["hops"]["floor90_db"], v["scores"]).verdict)
            today[f"{lab}|{var}"] = pd.Series(out).value_counts(normalize=True).round(3).to_dict()
    print("today (feature off):", json.dumps(today))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + "_today.json", "w") as f:
        json.dump(today, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=16)
    e = sub.add_parser("eval")
    e.add_argument("--noise-db", type=float, nargs="+", default=[3, 4.5, 6, 8, 10, 12, 15])
    e.add_argument("--abs-noise-db", type=float, nargs="+", default=[float("inf")])
    e.add_argument("--kind", choices=["joined", "long"], default=None)
    e.add_argument("--out", default=os.path.join(CACHE, "eval.csv"))
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

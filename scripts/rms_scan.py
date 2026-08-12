#!/usr/bin/env python
"""CLI: compute per-window RMS energy for a labeled WAV corpus, index-aligned with
offline_score.py's cached per-window model scores.

The permanent, documented way to backfill RMS onto an existing scored manifest for the
RMS-based verdict-confidence feature (see design/plan below) -- RMS is not part of
offline_score.py's own cache/manifest and deliberately stays that way: it's orders of
magnitude cheaper than TFLite inference and doesn't need to share a cache lifecycle
(SCORING_REV bump + full corpus rescore) with the model scores.

Design: docs/superpowers/specs/2026-08-09-rms-verdict-confidence-design.md
Plan:   docs/superpowers/plans/2026-08-09-rms-verdict-confidence.md

Backs the feature's core evidence: the RMS energy value main.py already computes for the
live energy bar (main.py:run_inference, `rms = np.sqrt(np.mean(audio_data**2))` on the
same 2.5s buffer fed to the model) but never feeds into the model carries information the
model's own features don't -- the model's log-mel features are floor-normalized per window
(see app/audio/features.py's 5th-percentile-floor step), which discards absolute loudness,
so RMS is not obviously redundant with what the model already sees. It turned out to be a
weak direct predictor of ground truth but a much stronger predictor of whether the model's
own verdict on a session is likely correct (see the design doc's Evidence section) --
that's what app/decision/rms_confidence.py's fitted curves consume, joined by `path` via
evaluate_decision_rules.py's --rms-manifest flag.

Reproduces score_wav_file's exact windowing (2.5s rolling buffer, 0.5s hop, held off
until the buffer fills once with real audio, capped at --max-duration-sec, whole blocks
only) so window index i here lines up 1:1 with window index i in an offline_score.py
score cache for the same file. Not model-dependent -- one RMS cache serves every model's
manifest.

Usage:
  <full-deps-python> scripts/rms_scan.py \
      --manifest manifest.csv \
      --cache-dir .rms_cache \
      --out rms_manifest.csv \
      --workers 8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

TARGET_SR = 44100
HOP_SEC = 0.5
WINDOW_SEC = 2.5
DEFAULT_MAX_DURATION_SEC = 20.0
# Bump if the windowing semantics change, so a cache from the old rules is never reused.
SCAN_REV = 1


def bandpass_filter(y, low_cut, high_cut, sr, order=4):
    """Matches app/audio/processor.py:AudioProcessor.bandpass_filter exactly: single
    Butterworth bandpass, one-pass lfilter (not filtfilt) -- same filter the app applies
    to the model's input buffer when the (off-by-default) bandpass setting is enabled."""
    nyquist = 0.5 * sr
    low = low_cut / nyquist
    high = high_cut / nyquist
    if high >= 1.0:
        high = 0.999
    b, a = butter(order, [low, high], btype="band", analog=False)
    return lfilter(b, a, y)


def _load_wav_mono(path):
    data, fs = sf.read(path, always_2d=True)
    mono = np.asarray(data)[:, 0].astype(np.float32)
    if fs != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=fs, target_sr=TARGET_SR).astype(np.float32)
    return mono


def _cache_key(path, max_duration_sec, bandpass):
    stat = os.stat(path)
    payload = json.dumps(
        {
            "path": os.path.abspath(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "scan_rev": SCAN_REV,
            "hop_sec": HOP_SEC,
            "window_sec": WINDOW_SEC,
            "max_duration_sec": max_duration_sec,
            "bandpass": bandpass,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rms_windows(path, max_duration_sec=DEFAULT_MAX_DURATION_SEC, bandpass=False,
                 low_cut=500.0, up_cut=8000.0):
    """Mirrors offline_score.py:score_wav_file's sliding-window branch exactly, but
    returns per-window RMS of the buffer instead of a model score. When bandpass=True,
    RMS is computed on the same 500-8000Hz-filtered buffer the app would feed the model
    with the (off-by-default) bandpass setting enabled -- matching
    app/audio/processor.py:AudioProcessor.process_audio's use_filter path."""
    audio = _load_wav_mono(path)

    block_size = int(TARGET_SR * HOP_SEC)
    buffer_len = int(TARGET_SR * WINDOW_SEC)

    available_hops = len(audio) // block_size
    n_hops = min(int(max_duration_sec / HOP_SEC), available_hops)

    buffer = np.zeros(buffer_len, dtype=np.float32)
    samples_received = 0
    values = []
    for hop in range(n_hops):
        start = hop * block_size
        chunk = audio[start:start + block_size]

        buffer = np.roll(buffer, -block_size)
        buffer[-block_size:] = chunk
        samples_received = min(buffer_len, samples_received + block_size)
        if samples_received < buffer_len:
            continue

        signal = buffer.astype(np.float64)
        if bandpass:
            signal = bandpass_filter(signal, low_cut, up_cut, TARGET_SR)
        values.append(float(np.sqrt(np.mean(signal ** 2))))

    return values


def _worker(args):
    path, cache_path, max_duration_sec, bandpass = args
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return path, json.load(f)["rms"]
    try:
        values = rms_windows(path, max_duration_sec, bandpass=bandpass)
    except Exception as e:  # noqa: BLE001 - corpus scan: log and skip, don't crash the run
        return path, {"__error__": str(e)}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"path": path, "rms": values}, f)
    return path, values


def run(args):
    os.makedirs(args.cache_dir, exist_ok=True)

    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    jobs = []
    for row in rows:
        path = row["path"]
        if not os.path.exists(path):
            continue
        key = _cache_key(path, args.max_duration_sec, args.bandpass)
        cache_path = os.path.join(args.cache_dir, f"{key}.json")
        jobs.append((row, path, cache_path))

    to_run = [(path, cache_path, args.max_duration_sec, args.bandpass) for _row, path, cache_path in jobs if not os.path.exists(cache_path)]
    print(f"{len(jobs)} files; {len(to_run)} need RMS scan ({len(jobs) - len(to_run)} cache hits)")

    if to_run:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_worker, job) for job in to_run]
                for i, fut in enumerate(as_completed(futures), 1):
                    fut.result()
                    if i % 200 == 0 or i == len(to_run):
                        print(f"  scanned {i}/{len(to_run)}")
        else:
            for i, job in enumerate(to_run, 1):
                _worker(job)
                if i % 200 == 0 or i == len(to_run):
                    print(f"  scanned {i}/{len(to_run)}")

    n_errors = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "month_bucket", "cache_path", "rms_cache_path", "n_rms_windows", "rms_json"])
        for row, path, cache_path in jobs:
            with open(cache_path, "r", encoding="utf-8") as cf:
                values = json.load(cf)["rms"]
            if isinstance(values, dict) and "__error__" in values:
                n_errors += 1
                continue
            writer.writerow(
                [path, row.get("label", ""), row.get("month_bucket", ""), row.get("cache_path", ""),
                 cache_path, len(values), json.dumps(values)]
            )

    print(f"Wrote {args.out} ({len(jobs) - n_errors} files, {n_errors} errors)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="offline_score.py-style manifest CSV (needs path/label/cache_path columns).")
    parser.add_argument("--cache-dir", default=".rms_cache")
    parser.add_argument("--out", default="rms_manifest.csv")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC)
    parser.add_argument("--bandpass", action="store_true",
                         help="Compute RMS on the 500-8000Hz-bandpassed buffer instead of raw broadband audio, matching the app's (off-by-default) bandpass setting.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

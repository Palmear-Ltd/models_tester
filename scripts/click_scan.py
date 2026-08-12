#!/usr/bin/env python
"""CLI: compute per-window click_count/click_rate (T009 ClickTransientCheck) for a
labeled WAV corpus, index-aligned with offline_score.py's cached per-window model scores.

Companion to scripts/rms_scan.py -- same windowing, same alignment guarantee. Where
rms_scan investigates whether RMS energy helps, this investigates whether the click-train
statistics app/health/checks/time_domain.py:ClickTransientCheck (T009) already computes
for a different purpose (flagging SENSOR_LINK contact noise) carry a direct RPW bite-click
signal -- run the exact production check via import rather than reimplementing it, so any
future threshold/algorithm change here is automatically picked up.

Usage:
  <full-deps-python> scripts/click_scan.py \
      --manifest refit_out/9_1_2_full_corpus/manifest.csv \
      --cache-dir .click_cache \
      --out click_manifest.csv \
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

from app.health.checks.time_domain import ClickTransientCheck
from app.health.models import AudioWindow

TARGET_SR = 44100
HOP_SEC = 0.5
WINDOW_SEC = 2.5
DEFAULT_MAX_DURATION_SEC = 20.0
SCAN_REV = 1

_CHECK = ClickTransientCheck()


def _load_wav_mono(path):
    data, fs = sf.read(path, always_2d=True)
    mono = np.asarray(data)[:, 0].astype(np.float32)
    if fs != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=fs, target_sr=TARGET_SR).astype(np.float32)
    return mono


def _cache_key(path, max_duration_sec):
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
            "click_k": _CHECK.click_k,
            "merge_gap": _CHECK.merge_gap,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def click_windows(path, max_duration_sec=DEFAULT_MAX_DURATION_SEC):
    """Mirrors offline_score.py:score_wav_file's sliding-window branch exactly, but
    returns per-window (click_count, click_rate) from the production T009 check instead
    of a model score."""
    audio = _load_wav_mono(path)

    block_size = int(TARGET_SR * HOP_SEC)
    buffer_len = int(TARGET_SR * WINDOW_SEC)

    available_hops = len(audio) // block_size
    n_hops = min(int(max_duration_sec / HOP_SEC), available_hops)

    buffer = np.zeros(buffer_len, dtype=np.float32)
    samples_received = 0
    counts = []
    rates = []
    for hop in range(n_hops):
        start = hop * block_size
        chunk = audio[start:start + block_size]

        buffer = np.roll(buffer, -block_size)
        buffer[-block_size:] = chunk
        samples_received = min(buffer_len, samples_received + block_size)
        if samples_received < buffer_len:
            continue

        window = AudioWindow(samples=buffer, sample_rate=TARGET_SR)
        result = _CHECK.run(window, {})
        by_name = {m.name: m.value for m in result.measurements}
        counts.append(float(by_name.get("click_count", 0.0)))
        rates.append(float(by_name.get("click_rate", 0.0)))

    return counts, rates


def _worker(args):
    path, cache_path, max_duration_sec = args
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return path, data["counts"], data["rates"]
    try:
        counts, rates = click_windows(path, max_duration_sec)
    except Exception as e:  # noqa: BLE001 - corpus scan: log and skip, don't crash the run
        return path, {"__error__": str(e)}, None
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"path": path, "counts": counts, "rates": rates}, f)
    return path, counts, rates


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
        key = _cache_key(path, args.max_duration_sec)
        cache_path = os.path.join(args.cache_dir, f"{key}.json")
        jobs.append((row, path, cache_path))

    to_run = [(path, cache_path, args.max_duration_sec) for _row, path, cache_path in jobs if not os.path.exists(cache_path)]
    print(f"{len(jobs)} files; {len(to_run)} need click scan ({len(jobs) - len(to_run)} cache hits)")

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
        writer.writerow(["path", "label", "month_bucket", "cache_path", "click_cache_path", "n_windows", "counts_json", "rates_json"])
        for row, path, cache_path in jobs:
            with open(cache_path, "r", encoding="utf-8") as cf:
                data = json.load(cf)
            if "__error__" in data:
                n_errors += 1
                continue
            counts = data["counts"]
            writer.writerow(
                [path, row.get("label", ""), row.get("month_bucket", ""), row.get("cache_path", ""),
                 cache_path, len(counts), json.dumps(counts), json.dumps(data["rates"])]
            )

    print(f"Wrote {args.out} ({len(jobs) - n_errors} files, {n_errors} errors)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default=".click_cache")
    parser.add_argument("--out", default="click_manifest.csv")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

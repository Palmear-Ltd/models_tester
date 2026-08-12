#!/usr/bin/env python
"""CLI: extract per-click normalized power spectra (fixed 40-bin, 0-8000Hz) + click
times for a list of WAV files, parallelized and cached per-file like rms_scan.py /
click_scan.py.

Companion to the click-template bite-vs-noise investigation
(docs/superpowers/specs -- see the RMS-verdict-confidence spec's sibling for this
feature once written). Detection reuses the same algorithm as
app/health/checks/time_domain.py:ClickTransientCheck (robust-sigma first-difference
threshold, click_k=8.0, merge_gap=3), run once over the whole (up-to-20s) file.

Usage:
  <full-deps-python> scripts/click_template_scan.py \
      --file-list paths.txt --cache-dir .click_template_cache \
      --out click_template_manifest.csv --workers 8
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

TARGET_SR = 44100
MAX_DURATION_SEC = 20.0
CLICK_K = 8.0
MERGE_GAP = 3
SNIPPET_MS = 30.0
N_BINS = 40
FREQ_MAX = 8000.0
SCAN_REV = 1


def load_wav_mono(path, max_dur=MAX_DURATION_SEC):
    data, fs = sf.read(path, always_2d=True)
    mono = np.asarray(data)[:, 0].astype(np.float64)
    if fs != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=fs, target_sr=TARGET_SR)
    n = int(TARGET_SR * max_dur)
    return mono[:n]


def detect_clicks(x):
    d = np.diff(x)
    if d.size == 0:
        return []
    sigma = 1.4826 * float(np.median(np.abs(d - np.median(d))))
    if sigma <= 0:
        return []
    mask = np.abs(d) > CLICK_K * sigma
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return []
    gaps = np.diff(idxs)
    group_starts = np.where(np.concatenate(([True], gaps > MERGE_GAP)))[0]
    group_bounds = list(zip(group_starts, list(group_starts[1:]) + [len(idxs)]))
    peaks = []
    for gs, ge in group_bounds:
        members = idxs[gs:ge]
        local_peak = members[np.argmax(np.abs(d[members]))]
        peaks.append(int(local_peak))
    return peaks


def click_spectrum(x, peak_idx, sr=TARGET_SR, half_ms=SNIPPET_MS, n_bins=N_BINS, freq_max=FREQ_MAX):
    half = int(sr * half_ms / 1000.0)
    lo, hi = peak_idx - half, peak_idx + half
    if lo < 0 or hi >= len(x):
        return None
    snip = x[lo:hi]
    n = len(snip)
    win = np.hamming(n)
    spec = np.abs(np.fft.rfft(snip * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    edges = np.linspace(0, freq_max, n_bins + 1)
    binned = np.zeros(n_bins)
    for i in range(n_bins):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        binned[i] = spec[m].sum() if m.any() else 0.0
    norm = np.linalg.norm(binned)
    if norm <= 0:
        return None
    return (binned / norm)


def extract_file(path, max_dur=MAX_DURATION_SEC):
    x = load_wav_mono(path, max_dur)
    peaks = detect_clicks(x)
    specs, times = [], []
    for pk in peaks:
        s = click_spectrum(x, pk)
        if s is not None:
            specs.append(s.tolist())
            times.append(pk / TARGET_SR)
    return specs, times


def _cache_key(path):
    stat = os.stat(path)
    payload = json.dumps(
        {"path": os.path.abspath(path), "size": stat.st_size, "mtime": stat.st_mtime,
         "scan_rev": SCAN_REV, "n_bins": N_BINS, "freq_max": FREQ_MAX,
         "snippet_ms": SNIPPET_MS, "click_k": CLICK_K, "merge_gap": MERGE_GAP},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _worker(args):
    path, cache_path = args
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            return path, d["specs"], d["times"]
    try:
        specs, times = extract_file(path)
    except Exception as e:  # noqa: BLE001
        return path, {"__error__": str(e)}, None
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"path": path, "specs": specs, "times": times}, f)
    return path, specs, times


def run(args):
    os.makedirs(args.cache_dir, exist_ok=True)
    with open(args.file_list, "r", encoding="utf-8") as f:
        paths = [l.strip() for l in f if l.strip()]

    jobs = [(p, os.path.join(args.cache_dir, f"{_cache_key(p)}.json")) for p in paths if os.path.exists(p)]
    to_run = [j for j in jobs if not os.path.exists(j[1])]
    print(f"{len(jobs)} files; {len(to_run)} need extraction ({len(jobs) - len(to_run)} cache hits)")

    if to_run:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_worker, job) for job in to_run]
                for i, fut in enumerate(as_completed(futures), 1):
                    fut.result()
                    if i % 40 == 0 or i == len(to_run):
                        print(f"  extracted {i}/{len(to_run)}")
        else:
            for i, job in enumerate(to_run, 1):
                _worker(job)
                if i % 40 == 0 or i == len(to_run):
                    print(f"  extracted {i}/{len(to_run)}")

    out_rows = []
    n_errors = 0
    for path, cache_path in jobs:
        with open(cache_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if "__error__" in d:
            n_errors += 1
            continue
        out_rows.append({"path": path, "n_clicks": len(d["specs"]), "cache_path": cache_path})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "n_clicks", "cache_path"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {args.out} ({len(out_rows)} files, {n_errors} errors)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--cache-dir", default=".click_template_cache")
    parser.add_argument("--out", default="click_template_manifest.csv")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

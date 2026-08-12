#!/usr/bin/env python
"""CLI: extract per-WINDOW click spectra + counts for a labeled WAV corpus, using the
REAL production rolling-buffer windowing (2.5s window, 0.5s hop, buffer-fill hold-off,
20s cap) instead of scripts/click_template_scan.py's single-pass whole-file detection.

This exists to close the windowing-parity gap flagged in
docs/superpowers/specs/2026-08-09-rootcause-click-template-refinement-design.md §3: the
differential spectral template investigated earlier in this session was validated against
single-pass whole-file click detection, NOT the per-window computation the live pipeline
(main.py:handle_audio_chunk / offline_score.py:score_wav_file) actually runs. Mirrors
scripts/rms_scan.py's windowing code exactly (same buffer-fill hold-off logic, same
hop/window/cap constants) and scripts/click_template_scan.py's detection/spectrum
algorithm (robust-MAD-sigma first-difference threshold, click_k=8.0/merge_gap=3; 40-bin
0-8000Hz L2-normalized power spectrum per click via a +/-30ms Hamming-windowed FFT).

Output: one JSON blob per file (cached, keyed like every other scan script in this repo)
with a list of per-window {"click_count": int, "specs": [[...40 floats...], ...]} entries
-- one list entry per window the live pipeline would actually analyze (i.e. NOT one entry
per hop; the leading hold-off hops are skipped entirely, exactly like rms_scan.py /
click_scan.py). A manifest CSV indexes the per-file cache paths for downstream fitting/
evaluation (fit_click_template.py).

Usage:
  <full-deps-python> scripts/click_template_scan_windowed.py \
      --manifest manifest.csv --cache-dir .click_template_windowed_cache \
      --out click_template_windowed_manifest.csv --workers 8
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
HOP_SEC = 0.5
WINDOW_SEC = 2.5
DEFAULT_MAX_DURATION_SEC = 20.0

CLICK_K = 8.0
MERGE_GAP = 3
SNIPPET_MS = 30.0
N_BINS = 40
FREQ_MAX = 8000.0

# Bump if windowing or click/spectrum semantics change, so a cache from the old rules is
# never silently reused.
SCAN_REV = 1


def _load_wav_mono(path):
    data, fs = sf.read(path, always_2d=True)
    mono = np.asarray(data)[:, 0].astype(np.float32)
    if fs != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=fs, target_sr=TARGET_SR).astype(np.float32)
    return mono


def detect_clicks(x: np.ndarray) -> list:
    """Same algorithm as app/health/checks/time_domain.py:ClickTransientCheck.run --
    duplicated (not imported) here deliberately, mirroring T010's own independence from
    T009 (checks stay independent; this scan script exists to validate what becomes T010,
    not to reuse T009's plumbing). Returns a list of peak sample indices, one per merged
    click group, local to the array `x` passed in (a single window buffer)."""
    d = np.diff(x.astype(np.float64))
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
    return binned / norm


def click_windows_and_spectra(path, max_duration_sec=DEFAULT_MAX_DURATION_SEC):
    """Reproduces offline_score.py:score_wav_file's / scripts/rms_scan.py's sliding-window
    branch exactly: a persistent 2.5s buffer, zero-initialized, rolled+updated every 0.5s
    hop, with no window yielded until the buffer has filled once with real audio. Runs
    click detection + spectrum extraction independently on EACH window's full buffer (a
    physical click near a window boundary is therefore seen -- and its spectrum extracted
    -- in multiple consecutive overlapping windows, same redundancy T009's own click_count
    already has in production; see the design spec §3)."""
    audio = _load_wav_mono(path)

    block_size = int(TARGET_SR * HOP_SEC)
    buffer_len = int(TARGET_SR * WINDOW_SEC)

    available_hops = len(audio) // block_size
    n_hops = min(int(max_duration_sec / HOP_SEC), available_hops)

    buffer = np.zeros(buffer_len, dtype=np.float32)
    samples_received = 0
    windows = []
    for hop in range(n_hops):
        start = hop * block_size
        chunk = audio[start:start + block_size]

        buffer = np.roll(buffer, -block_size)
        buffer[-block_size:] = chunk
        samples_received = min(buffer_len, samples_received + block_size)
        if samples_received < buffer_len:
            continue

        peaks = detect_clicks(buffer)
        specs = []
        for pk in peaks:
            s = click_spectrum(buffer, pk)
            if s is not None:
                specs.append(s.tolist())
        windows.append({"click_count": len(peaks), "specs": specs})

    return windows


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
            "click_k": CLICK_K,
            "merge_gap": MERGE_GAP,
            "n_bins": N_BINS,
            "freq_max": FREQ_MAX,
            "snippet_ms": SNIPPET_MS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _worker(args):
    path, cache_path, max_duration_sec = args
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return path, json.load(f)["windows"]
    try:
        windows = click_windows_and_spectra(path, max_duration_sec)
    except Exception as e:  # noqa: BLE001 - corpus scan: log and skip, don't crash the run
        return path, {"__error__": str(e)}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"path": path, "windows": windows}, f)
    return path, windows


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
    print(f"{len(jobs)} files; {len(to_run)} need windowed click scan ({len(jobs) - len(to_run)} cache hits)")

    if to_run:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_worker, job) for job in to_run]
                for i, fut in enumerate(as_completed(futures), 1):
                    fut.result()
                    if i % 50 == 0 or i == len(to_run):
                        print(f"  scanned {i}/{len(to_run)}")
        else:
            for i, job in enumerate(to_run, 1):
                _worker(job)
                if i % 50 == 0 or i == len(to_run):
                    print(f"  scanned {i}/{len(to_run)}")

    n_errors = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "month_bucket", "n_windows", "total_clicks", "cache_path"])
        for row, path, cache_path in jobs:
            with open(cache_path, "r", encoding="utf-8") as cf:
                data = json.load(cf)
            windows = data["windows"]
            if isinstance(windows, dict) and "__error__" in windows:
                n_errors += 1
                continue
            total_clicks = sum(w["click_count"] for w in windows)
            writer.writerow(
                [path, row.get("label", ""), row.get("month_bucket", ""), len(windows), total_clicks, cache_path]
            )

    print(f"Wrote {args.out} ({len(jobs) - n_errors} files, {n_errors} errors)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="offline_score.py-style manifest CSV (needs path/label/month_bucket columns).")
    parser.add_argument("--cache-dir", default=".click_template_windowed_cache")
    parser.add_argument("--out", default="click_template_windowed_manifest.csv")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

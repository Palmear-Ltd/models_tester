#!/usr/bin/env python
"""CLI: fit + evaluate a locally-fit differential click spectral template for the T010
check proposed in docs/superpowers/specs/2026-08-09-rootcause-click-template-refinement-
design.md, and run this refinement's Phase A "decision gate" (see the plan of the same
date): does the AUC lift this session found (0.71-0.72, T vs FAULT-gold) survive when the
click spectra are extracted using the REAL production per-window rolling-buffer windowing
instead of the single-pass whole-file detection that produced those numbers?

Two-stage pipeline, mirroring calibrate.py's split (WAV/soundfile I/O stays out of
app/health/):
  1. scripts/click_template_scan_windowed.py extracts, per labeled WAV file, a list of
     per-window {click_count, specs} blobs using the real 2.5s/0.5s-hop/hold-off/20s-cap
     windowing (cached to disk). This script imports and drives that module directly
     rather than shelling out, so one invocation does the whole pipeline.
  2. This script selects train/test files (stratified by label, 2021 excluded --
     deliberately-injected noise that year per this session's prior turns -- seed=42),
     pools per-window click spectra into a differential template (mean T-train click
     spectrum minus mean TN-train click spectrum, L2-normalized), fits a per-click match
     threshold via Youden's J on the pooled train clicks, then evaluates the resulting
     per-file `matched_click_count` (sum across a session's windows, mirroring how
     rootcause.py's assess_many already aggregates T009's per-window score today) against
     held-out T-test / TN-test / FAULT-gold (test_data/audio_signal_health/fp/F, confirmed
     SENSOR_LINK faults) / FP-silver (model-flagged-but-unconfirmed FPs, sampled from
     9_1_4/audio_data/wav/F's FP_-prefixed files) -- alongside the raw click_count baseline
     for comparison.

Only writes app/health/click_template.json if invoked with --write-output; by default this
is a read-only evaluation run so the decision gate can be reviewed before anything ships.

Usage:
  <full-deps-python> fit_click_template.py \
      --manifest manifest.csv --workers 8 \
      [--write-output --output app/health/click_template.json]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:  # pragma: no cover - sklearn is a declared project dependency
    roc_auc_score = None
    roc_curve = None

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
import click_template_scan_windowed as ctw  # noqa: E402

_9_1_4_DEFAULT = os.path.expanduser("/home/bashar/workspace/palmear/9_1_4")


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------


def _load_manifest_rows(manifest_path):
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") not in ("T", "F"):
                continue
            if row.get("ambiguous_prefix") in ("True", "1", "true"):
                continue
            month = row.get("month_bucket") or ""
            if month.startswith("2021"):
                continue  # deliberately-injected noise that year (prior turns)
            if not os.path.exists(row["path"]):
                continue
            rows.append(row)
    return rows


def _basename(path):
    return os.path.basename(path)


def select_pools(manifest_path, fp_silver_root):
    rows = _load_manifest_rows(manifest_path)
    t_pool, tn_pool, fp_silver_pool = [], [], []
    fp_silver_root_norm = os.path.normpath(fp_silver_root)
    for row in rows:
        path = row["path"]
        label = row["label"]
        if label == "T":
            t_pool.append(row)
        else:  # "F"
            is_fp_silver = (
                _basename(path).startswith("FP_")
                and os.path.normpath(path).startswith(fp_silver_root_norm)
            )
            if is_fp_silver:
                fp_silver_pool.append(row)
            else:
                tn_pool.append(row)
    return t_pool, tn_pool, fp_silver_pool


def stratified_split(rows, test_frac=0.3, seed=42):
    """Same idiom as evaluate_decision_rules.py:stratified_split -- deterministic
    per-group shuffle+split so fitting and evaluation never share files."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_test = max(1, int(len(rows) * test_frac)) if len(rows) > 1 else 0
    return rows[n_test:], rows[:n_test]  # train, test


def sample(rows, n, seed=42):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------------------
# Windowed scan (drives scripts/click_template_scan_windowed.py directly)
# ---------------------------------------------------------------------------


def scan_files(paths, cache_dir, max_duration_sec, workers):
    os.makedirs(cache_dir, exist_ok=True)
    jobs = []
    for path in paths:
        key = ctw._cache_key(path, max_duration_sec)
        cache_path = os.path.join(cache_dir, f"{key}.json")
        jobs.append((path, cache_path, max_duration_sec))

    to_run = [j for j in jobs if not os.path.exists(j[1])]
    print(f"  {len(jobs)} files; {len(to_run)} need windowed click scan ({len(jobs) - len(to_run)} cache hits)")

    if to_run:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(ctw._worker, job) for job in to_run]
                for i, fut in enumerate(as_completed(futures), 1):
                    fut.result()
                    if i % 50 == 0 or i == len(to_run):
                        print(f"    scanned {i}/{len(to_run)}")
        else:
            for i, job in enumerate(to_run, 1):
                ctw._worker(job)
                if i % 50 == 0 or i == len(to_run):
                    print(f"    scanned {i}/{len(to_run)}")

    results = {}
    n_errors = 0
    for path, cache_path, _ in jobs:
        with open(cache_path, "r", encoding="utf-8") as f:
            windows = json.load(f)["windows"]
        if isinstance(windows, dict) and "__error__" in windows:
            n_errors += 1
            continue
        results[path] = windows
    if n_errors:
        print(f"  ({n_errors} files errored during scan and were skipped)")
    return results


# ---------------------------------------------------------------------------
# Template fitting
# ---------------------------------------------------------------------------


def pooled_click_specs(paths, scanned):
    """All per-click spectra across all windows across all files in `paths`."""
    specs = []
    for path in paths:
        for w in scanned.get(path, []):
            specs.extend(w["specs"])
    return specs


def fit_template(train_t_paths, train_tn_paths, scanned):
    t_specs = np.array(pooled_click_specs(train_t_paths, scanned))
    tn_specs = np.array(pooled_click_specs(train_tn_paths, scanned))
    if t_specs.size == 0 or tn_specs.size == 0:
        raise RuntimeError(
            f"Not enough clicks to fit a template (T train clicks={len(t_specs)}, "
            f"TN train clicks={len(tn_specs)}). Corpus too small or click detection "
            "found nothing in the train split."
        )

    diff = t_specs.mean(axis=0) - tn_specs.mean(axis=0)
    norm = np.linalg.norm(diff)
    template = diff / norm if norm > 0 else diff

    all_specs = np.concatenate([t_specs, tn_specs], axis=0)
    all_labels = np.concatenate([np.ones(len(t_specs)), np.zeros(len(tn_specs))])
    scores = all_specs @ template

    threshold = _youden_threshold(all_labels, scores)
    return template, threshold, len(t_specs), len(tn_specs)


def _youden_threshold(labels, scores, default=0.0):
    if roc_curve is not None and len(set(labels.tolist())) >= 2:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        best = int(np.argmax(tpr - fpr))
        return float(thresholds[best])
    # Pure-NumPy fallback (project convention, see evaluate_decision_rules.py): sweep
    # every observed score as a candidate cutoff.
    candidates = np.unique(scores)
    best_j, best_t = -1.0, default
    for t in candidates:
        pred = scores >= t
        tp = np.sum(pred & (labels == 1))
        fn = np.sum(~pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        tn = np.sum(~pred & (labels == 0))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Per-file statistics + AUC evaluation
# ---------------------------------------------------------------------------


def file_stats(paths, scanned, template, threshold):
    """Returns {path: (raw_click_count, matched_click_count)} -- both summed across a
    session's windows, mirroring rootcause.py:assess_many's sum-then-mean aggregation of
    a per-window measurement (see design spec §3)."""
    out = {}
    for path in paths:
        raw = 0
        matched = 0
        for w in scanned.get(path, []):
            raw += w["click_count"]
            for spec in w["specs"]:
                score = float(np.dot(spec, template))
                if score >= threshold:
                    matched += 1
        out[path] = (raw, matched)
    return out


def _auc(pos_values, neg_values):
    if roc_auc_score is not None:
        if len(pos_values) == 0 or len(neg_values) == 0:
            return float("nan")
        y = [1] * len(pos_values) + [0] * len(neg_values)
        s = list(pos_values) + list(neg_values)
        if len(set(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    # Mann-Whitney U / rank-AUC fallback.
    if not pos_values or not neg_values:
        return float("nan")
    all_vals = sorted(pos_values + neg_values)
    ranks = {v: i + 1 for i, v in enumerate(all_vals)}  # ties not de-duplicated; fine for a fallback
    rank_sum_pos = sum(ranks[v] for v in pos_values)
    n_pos, n_neg = len(pos_values), len(neg_values)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def report_comparison(name, a_stats, b_stats, index):
    """index: 0 for raw_click_count, 1 for matched_click_count."""
    a_vals = [v[index] for v in a_stats.values()]
    b_vals = [v[index] for v in b_stats.values()]
    return _auc(a_vals, b_vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args):
    print(f"Selecting files from {args.manifest} (2021 excluded, seed={args.seed}) ...")
    t_pool, tn_pool, fp_silver_pool = select_pools(args.manifest, args.fp_silver_root)
    print(f"  candidate pools: T={len(t_pool)} TN={len(tn_pool)} FP-silver={len(fp_silver_pool)}")

    t_train_rows, t_test_rows = stratified_split(t_pool, seed=args.seed)
    tn_train_rows, tn_test_rows = stratified_split(tn_pool, seed=args.seed)

    t_train_rows = t_train_rows[: args.train_cap]
    tn_train_rows = tn_train_rows[: args.train_cap]
    t_test_rows = t_test_rows[: args.test_cap]
    tn_test_rows = tn_test_rows[: args.test_cap]
    fp_silver_rows = sample(fp_silver_pool, args.fp_silver_cap, seed=args.seed)

    fault_gold_paths = sorted(glob.glob(os.path.join(args.fault_gold_dir, "*.wav")))
    local_tn_paths = sorted(glob.glob(os.path.join(args.local_tn_dir, "*.wav")))

    t_train = [r["path"] for r in t_train_rows]
    tn_train = [r["path"] for r in tn_train_rows]
    t_test = [r["path"] for r in t_test_rows]
    tn_test = [r["path"] for r in tn_test_rows]
    fp_silver = [r["path"] for r in fp_silver_rows]

    print(
        f"  split: T train={len(t_train)} test={len(t_test)} | "
        f"TN train={len(tn_train)} test={len(tn_test)} | "
        f"FP-silver eval={len(fp_silver)} | FAULT-gold={len(fault_gold_paths)} | "
        f"local TN ref={len(local_tn_paths)}"
    )

    all_paths = sorted(set(t_train + tn_train + t_test + tn_test + fp_silver + fault_gold_paths + local_tn_paths))
    print(f"Windowed click scan over {len(all_paths)} files (real production windowing) ...")
    scanned = scan_files(all_paths, args.cache_dir, args.max_duration_sec, args.workers)

    print("Fitting differential template on train split ...")
    template, threshold, n_t_clicks, n_tn_clicks = fit_template(t_train, tn_train, scanned)
    print(f"  train clicks pooled: T={n_t_clicks} TN={n_tn_clicks}; match_threshold={threshold:.4f}")

    print("\nComputing per-file raw_click_count / matched_click_count for held-out sets ...")
    stats = {
        "t_test": file_stats(t_test, scanned, template, threshold),
        "tn_test": file_stats(tn_test, scanned, template, threshold),
        "fault_gold": file_stats(fault_gold_paths, scanned, template, threshold),
        "fp_silver": file_stats(fp_silver, scanned, template, threshold),
        "local_tn_ref": file_stats(local_tn_paths, scanned, template, threshold),
    }

    print("\n=== Held-out AUC (rank-AUC of session statistic, positive=first group) ===")
    print(f"{'comparison':45s} {'raw click AUC':>14s} {'matched click AUC':>18s}")
    comparisons = [
        ("T-test vs FAULT-gold", "t_test", "fault_gold"),
        ("T-test vs FP-silver", "t_test", "fp_silver"),
        ("TN-test vs FAULT-gold", "tn_test", "fault_gold"),
        ("TN-test vs FP-silver", "tn_test", "fp_silver"),
    ]
    results = {}
    for name, a_key, b_key in comparisons:
        raw_auc = report_comparison(name, stats[a_key], stats[b_key], 0)
        matched_auc = report_comparison(name, stats[a_key], stats[b_key], 1)
        results[name] = (raw_auc, matched_auc)
        print(f"{name:45s} {raw_auc:14.3f} {matched_auc:18.3f}")

    gate_raw, gate_matched = results["T-test vs FAULT-gold"]
    print("\n=== Phase A decision gate (T-test vs FAULT-gold, matched_click_count) ===")
    print(f"  raw click count AUC:      {gate_raw:.3f}")
    print(f"  matched click count AUC:  {gate_matched:.3f}")
    print("  this session's single-pass exploration found: raw~0.50, matched~0.58, burst-rate 0.71-0.72")
    lift = gate_matched - gate_raw
    survives = gate_matched >= 0.60 and lift >= 0.05
    print(f"  lift over raw baseline:   {lift:+.3f}")
    print(f"  GATE: {'PASS -- proceed to wiring' if survives else 'FAIL -- do not wire into rootcause.py'}")

    if args.write_output:
        if not survives and not args.force_write:
            print(
                "\nRefusing to write click_template.json: the decision gate did not pass. "
                "Pass --force-write to write anyway (e.g. for inspection)."
            )
        else:
            payload = {
                "n_bins": ctw.N_BINS,
                "freq_max_hz": ctw.FREQ_MAX,
                "template": template.tolist(),
                "match_threshold": threshold,
                "fitted_against": (
                    f"T train n={len(t_train)} ({n_t_clicks} clicks) / "
                    f"TN train n={len(tn_train)} ({n_tn_clicks} clicks), "
                    f"9_1_4 + test_data corpora, 2021 excluded, seed={args.seed}, "
                    "real per-window production windowing (2.5s/0.5s hop, hold-off, 20s cap)"
                ),
                "fitted_date": date.today().isoformat(),
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\nWrote {args.output}")

    return results, stats, template, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--cache-dir", default=".click_template_windowed_cache")
    parser.add_argument("--fault-gold-dir", default="test_data/audio_signal_health/fp/F")
    parser.add_argument("--local-tn-dir", default="test_data/F")
    parser.add_argument("--fp-silver-root", default=os.path.join(_9_1_4_DEFAULT, "audio_data", "wav", "F"))
    parser.add_argument("--train-cap", type=int, default=183)
    parser.add_argument("--test-cap", type=int, default=150)
    parser.add_argument("--fp-silver-cap", type=int, default=150)
    parser.add_argument("--max-duration-sec", type=float, default=ctw.DEFAULT_MAX_DURATION_SEC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--write-output", action="store_true")
    parser.add_argument("--force-write", action="store_true")
    parser.add_argument("--output", default=os.path.join("app", "health", "click_template.json"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

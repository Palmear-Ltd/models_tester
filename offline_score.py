#!/usr/bin/env python
"""CLI: batch-score labeled WAV corpora into per-window probability sequences.

Offline counterpart to main.py's live sliding-window scoring loop
(main.py:834-873 `handle_audio_chunk`, main.py:588-622 `file_loop`) — replicates it
exactly (2.5s rolling buffer, 0.5s hop, zero-padded tail chunk) so a decision rule
validated against this script's output transfers unmodified to the live app. WAV I/O
(soundfile/librosa) and TFLite inference live here, not in app/decision, mirroring
calibrate.py's split between root-level I/O scripts and the portable app/ package.

Usage:
  <full-deps-python> offline_score.py \
      --corpus test_data \
      --corpus /path/to/external/corpus \
      --model models/9_1_2/model.tflite --scaler models/9_1_2/scaler.json \
      --cache-dir .score_cache --manifest-out manifest.csv

Assumes a sliding-window (non-one-shot) model, matching main.py's default behavior for
models/9_1_2. Ground truth is resolved from a `T`/`F` path component (see
app.decision.manifest.resolve_label) — files with no such component, or an ambiguous
`X_` filename prefix, are still scored but flagged for exclusion in the manifest rather
than silently trusted.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import soundfile as sf

from app.audio.features import FeatureExtractor
from app.audio.scaler import Scaler
from app.decision.manifest import is_ambiguous_prefix, parse_recording_date, resolve_label, season_bucket
from app.model.inference import ModelInference

TARGET_SR = 44100
HOP_SEC = 0.5
WINDOW_SEC = 2.5

# Must match main.py's Tk-var defaults exactly (main.py:970-982).
DEFAULT_PREP_PARAMS = dict(
    low_cut=500.0,
    up_cut=8000.0,
    fmin=50.0,
    fmax=10000.0,
    sub_win_size_sec=0.05,
    sub_hop_size_sec=0.025,
    use_filter=True,
)

MANIFEST_FIELDS = [
    "path",
    "label",
    "date",
    "month_bucket",
    "ambiguous_prefix",
    "cache_path",
    "corpus_root",
    "n_windows",
]


def _gather_wavs(root):
    matches = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                matches.append(os.path.join(dirpath, name))
    return sorted(matches)


def _load_wav_mono(path):
    data, fs = sf.read(path, always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    if fs != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=fs, target_sr=TARGET_SR).astype(np.float32)
    return mono


def _cache_key(path, model_path, scaler_path, prep_params):
    stat = os.stat(path)
    payload = json.dumps(
        {
            "path": os.path.abspath(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "model_path": os.path.abspath(model_path),
            "scaler_path": os.path.abspath(scaler_path),
            "prep_params": prep_params,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_wav_file(path, model, scaler, feature_extractor, seq_len, n_mels, prep_params):
    """Reproduces main.py's sliding-window scoring exactly: a 2.5s rolling buffer
    (zero-initialized, like a freshly started session), updated every 0.5s hop via
    `np.roll`, one score per hop. The leading windows are intentionally built from a
    mostly-zero buffer, same as a live session's first few seconds."""
    audio = _load_wav_mono(path)
    block_size = int(TARGET_SR * HOP_SEC)
    buffer_len = int(TARGET_SR * WINDOW_SEC)
    total_samples = len(audio)
    n_hops = math.ceil(total_samples / block_size) if total_samples > 0 else 0

    buffer = np.zeros(buffer_len, dtype=np.float32)
    scores = []
    for hop in range(n_hops):
        start = hop * block_size
        end = min(start + block_size, total_samples)
        chunk = audio[start:end]
        if len(chunk) < block_size:
            chunk = np.pad(chunk, (0, block_size - len(chunk)))

        buffer = np.roll(buffer, -block_size)
        buffer[-block_size:] = chunk

        specs = feature_extractor.extract_features(
            buffer, sr=TARGET_SR, n_mels=n_mels, seq_len=seq_len, **prep_params
        )
        specs_scaled = scaler.apply(specs)
        input_data = specs_scaled.reshape(1, seq_len, n_mels, 1).astype(np.float32)
        output = model.predict(input_data)
        score = float(output[0][0]) if output.shape[-1] == 1 else float(output[0][1])
        scores.append(score)

    return scores


# --- Worker process state (one ModelInference/Scaler per process; TFLite interpreters
# aren't fork/thread-shareable) ---
_worker_model = None
_worker_scaler = None
_worker_feature_extractor = None
_worker_seq_len = None
_worker_n_mels = None
_worker_prep_params = None


def _init_worker(model_path, scaler_path, prep_params):
    global _worker_model, _worker_scaler, _worker_feature_extractor
    global _worker_seq_len, _worker_n_mels, _worker_prep_params

    _worker_model = ModelInference()
    _worker_model.load_model(model_path)
    input_shape = _worker_model.get_input_shape()
    _worker_seq_len = int(input_shape[1])
    _worker_n_mels = int(input_shape[2])

    _worker_scaler = Scaler()
    mean, _var = _worker_scaler.load(scaler_path)
    if mean is None:
        raise RuntimeError(f"Failed to load scaler {scaler_path}: {_worker_scaler.last_error}")

    _worker_feature_extractor = FeatureExtractor()
    _worker_prep_params = prep_params


def _score_one(path):
    scores = score_wav_file(
        path,
        _worker_model,
        _worker_scaler,
        _worker_feature_extractor,
        _worker_seq_len,
        _worker_n_mels,
        _worker_prep_params,
    )
    return path, scores


def _write_cache(cache_path, path, scores):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"path": path, "scores": scores}, f)


def _read_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(args):
    prep_params = dict(DEFAULT_PREP_PARAMS)
    os.makedirs(args.cache_dir, exist_ok=True)

    all_wavs = []
    for root in args.corpus:
        found = _gather_wavs(root)
        if args.glob:
            found = [p for p in found if fnmatch.fnmatch(os.path.basename(p), args.glob)]
        all_wavs.extend((root, p) for p in found)

    if args.limit:
        all_wavs = all_wavs[: args.limit]

    manifest_rows = []
    to_score = []
    excluded_ambiguous = 0
    excluded_no_label = 0

    for corpus_root, path in all_wavs:
        base = os.path.basename(path)
        label = resolve_label(path)
        ambiguous = is_ambiguous_prefix(base)
        parsed_date = parse_recording_date(base)
        month = season_bucket(parsed_date) if parsed_date else None

        if ambiguous:
            excluded_ambiguous += 1
        if label is None:
            excluded_no_label += 1

        key = _cache_key(path, args.model, args.scaler, prep_params)
        cache_path = os.path.join(args.cache_dir, f"{key}.json")
        if not os.path.exists(cache_path):
            to_score.append((path, cache_path))

        manifest_rows.append(
            {
                "path": path,
                "label": label or "",
                "date": parsed_date.isoformat() if parsed_date else "",
                "month_bucket": month or "",
                "ambiguous_prefix": ambiguous,
                "cache_path": cache_path,
                "corpus_root": corpus_root,
                "n_windows": 0,
            }
        )

    print(
        f"{len(all_wavs)} files found; {len(to_score)} need scoring "
        f"({len(all_wavs) - len(to_score)} cache hits); "
        f"{excluded_ambiguous} ambiguous-prefix, {excluded_no_label} unresolved-label"
    )

    if to_score:
        if args.workers > 1:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(args.model, args.scaler, prep_params),
            ) as executor:
                futures = {
                    executor.submit(_score_one, path): (path, cache_path)
                    for path, cache_path in to_score
                }
                for i, future in enumerate(as_completed(futures), 1):
                    path, cache_path = futures[future]
                    _, scores = future.result()
                    _write_cache(cache_path, path, scores)
                    if i % 100 == 0 or i == len(to_score):
                        print(f"  scored {i}/{len(to_score)}")
        else:
            _init_worker(args.model, args.scaler, prep_params)
            for i, (path, cache_path) in enumerate(to_score, 1):
                _, scores = _score_one(path)
                _write_cache(cache_path, path, scores)
                if i % 100 == 0 or i == len(to_score):
                    print(f"  scored {i}/{len(to_score)}")

    for row in manifest_rows:
        cached = _read_cache(row["cache_path"])
        row["n_windows"] = len(cached["scores"]) if cached else 0

    with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote manifest: {args.manifest_out} ({len(manifest_rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-score labeled WAV corpora into per-window probability sequences "
            "(offline counterpart to main.py's live sliding-window loop)."
        )
    )
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        help="Root folder to walk recursively for .wav files (repeatable). Ground truth "
        "is resolved from a T/F path component under each root.",
    )
    parser.add_argument("--model", default=os.path.join("models", "9_1_2", "model.tflite"))
    parser.add_argument("--scaler", default=os.path.join("models", "9_1_2", "scaler.json"))
    parser.add_argument("--cache-dir", default=".score_cache")
    parser.add_argument("--manifest-out", default="manifest.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N files found (smoke runs).")
    parser.add_argument("--glob", default=None, help="Only include files whose basename matches this glob pattern.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

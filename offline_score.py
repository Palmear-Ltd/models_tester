#!/usr/bin/env python
"""CLI: batch-score labeled WAV corpora into per-window probability sequences.

Offline counterpart to main.py's live sliding-window scoring loop
(`handle_audio_chunk`, `file_loop`, `mic_loop`) — replicates it exactly so a decision
rule validated against this script's output transfers unmodified to the live app:

  - 2.5s rolling buffer, 0.5s hop, advanced via np.roll
  - held until the buffer has filled once with real audio (the leading hops are built
    from a still-partly-zero buffer, so they are skipped rather than scored)
  - channel 0 only, never a mean downmix -- mic_loop opens InputStream(channels=1)
  - whole 0.5s blocks only; a short trailing block is DROPPED, not zero-padded, because
    a padded tail puts a hard zero edge into the final windows that no live session sees
  - capped at --max-duration-sec (default 20s, matching main.py's sliding_test_duration_var)

Those last three are load-bearing, not cosmetic: EwmaPeakDecision.peak is a running max
over the session, so scoring more windows here than a live session produces would fit a
cutoff that the app can then cross more easily than the fit implied. Keep this file,
main.py:file_loop and main.py:mic_loop in lockstep -- see tests/test_offline_score_parity.py
and tests/test_input_path_parity.py.

WAV I/O (soundfile/librosa) and TFLite inference live here, not in app/decision,
mirroring calibrate.py's split between root-level I/O scripts and the portable app/
package.

Note: files shorter than WINDOW_SEC (2.5s) yield zero scores -- the buffer never fills
with real audio, so no hop is ever scored, matching a live session that ends before its
first 2.5s buffer fill completes.

Usage:
  <full-deps-python> offline_score.py \
      --corpus test_data \
      --corpus /path/to/external/corpus \
      --model models/9_1_2/model.tflite --scaler models/9_1_2/scaler.json \
      --cache-dir .score_cache --manifest-out manifest.csv

One-shot models (seq_len >= 784, or "one_shot" in the model path -- same heuristic as
main.py:load_model) are detected automatically and scored via a separate path that
mirrors main.py:run_single_shot_inference instead: no rolling buffer, a single inference
on the (zero-padded-or-truncated) first --max-duration-sec of audio, yielding exactly one
score. Applying the sliding-window loop to a one-shot model would feed it a 2.5s buffer
padded out to its full seq_len, which is not what a live single-shot session ever does.

Ground truth is resolved from a `T`/`F` path component (see
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
# Matches main.py's sliding_test_duration_var default (main.py:__init__). A live test
# auto-stops here, so scoring past it would fit the cutoff on windows the app never sees.
DEFAULT_MAX_DURATION_SEC = 20.0
# Bumped whenever the windowing semantics above change, so a cache written under the old
# rules is never silently reused by a refit (rev 2: channel 0, no padded tail, capped).
SCORING_REV = 2

# Must match main.py's Tk-var defaults exactly (main.py:970-982).
# use_filter=False: the mobile app ships with the bandpass filter disabled
# (opt-in via app settings), and most users never enable it.
DEFAULT_PREP_PARAMS = dict(
    low_cut=500.0,
    up_cut=8000.0,
    fmin=50.0,
    fmax=10000.0,
    sub_win_size_sec=0.05,
    sub_hop_size_sec=0.025,
    use_filter=False,
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
    """Channel 0 at TARGET_SR. Not a mean downmix: main.py:mic_loop opens the stream
    with channels=1, which yields the first channel, and file_loop matches it."""
    data, fs = sf.read(path, always_2d=True)
    mono = np.asarray(data)[:, 0].astype(np.float32)
    if fs != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=fs, target_sr=TARGET_SR).astype(np.float32)
    return mono


def _cache_key(path, model_path, scaler_path, prep_params, max_duration_sec=DEFAULT_MAX_DURATION_SEC):
    stat = os.stat(path)
    payload = json.dumps(
        {
            "path": os.path.abspath(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "model_path": os.path.abspath(model_path),
            "scaler_path": os.path.abspath(scaler_path),
            "prep_params": prep_params,
            # Windowing semantics are part of the identity of a cached score sequence.
            "scoring_rev": SCORING_REV,
            "hop_sec": HOP_SEC,
            "window_sec": WINDOW_SEC,
            "max_duration_sec": max_duration_sec,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_one_shot_model(model_path, seq_len):
    """Mirrors main.py:load_model's heuristic (main.py:510) for detecting a one-shot
    (single full-clip inference) model vs. a sliding-window one."""
    return seq_len >= 784 or "one_shot" in model_path.lower()


def _infer_score(buffer, model, scaler, feature_extractor, seq_len, n_mels, prep_params):
    specs = feature_extractor.extract_features(
        buffer, sr=TARGET_SR, n_mels=n_mels, seq_len=seq_len, **prep_params
    )
    specs_scaled = scaler.apply(specs)
    input_data = specs_scaled.reshape(1, seq_len, n_mels, 1).astype(np.float32)
    output = model.predict(input_data)
    return float(output[0][0]) if output.shape[-1] == 1 else float(output[0][1])


def _score_one_shot(audio, model, scaler, feature_extractor, seq_len, n_mels, prep_params, max_duration_sec):
    """Reproduces main.py:run_single_shot_inference: the full captured clip, zero-padded
    if shorter than max_duration_sec or truncated to it if longer, scored in a single
    inference -- no rolling buffer, no per-hop scores. An empty file yields no score,
    matching a session that captures no audio to infer on.

    max_duration_sec doubles here as main.py's separate single_shot_duration_sec (both
    default to 20s in main.py's __init__); pass --max-duration-sec accordingly if a
    one-shot model's Settings -> Single-shot Duration has been changed independently."""
    if len(audio) == 0:
        return []
    target_samples = int(TARGET_SR * max_duration_sec)
    if len(audio) < target_samples:
        clip = np.pad(audio, (0, target_samples - len(audio)))
    else:
        clip = audio[:target_samples]
    return [_infer_score(clip, model, scaler, feature_extractor, seq_len, n_mels, prep_params)]


def score_wav_file(
    path, model, scaler, feature_extractor, seq_len, n_mels, prep_params,
    max_duration_sec=DEFAULT_MAX_DURATION_SEC, model_path="",
):
    """Reproduces main.py's scoring exactly. Two branches, dispatched by model type
    (see _is_one_shot_model):

    Sliding-window (the default): a 2.5s rolling buffer (zero-initialized, like a
    freshly started session), updated every 0.5s hop via `np.roll`. Scoring is held
    until the buffer has filled once with real audio -- mirroring main.py's
    handle_audio_chunk buffer-fill hold-off -- so the leading hops (built from a
    still-partly-zero buffer) are skipped rather than scored. Whole blocks only, capped
    at max_duration_sec, matching main.py:file_loop: a short trailing block is dropped
    rather than zero-padded, and audio past the live capture duration is never scored.

    One-shot: see _score_one_shot."""
    audio = _load_wav_mono(path)

    if _is_one_shot_model(model_path, seq_len):
        return _score_one_shot(audio, model, scaler, feature_extractor, seq_len, n_mels, prep_params, max_duration_sec)

    block_size = int(TARGET_SR * HOP_SEC)
    buffer_len = int(TARGET_SR * WINDOW_SEC)

    available_hops = len(audio) // block_size
    n_hops = min(int(max_duration_sec / HOP_SEC), available_hops)

    buffer = np.zeros(buffer_len, dtype=np.float32)
    samples_received = 0
    scores = []
    for hop in range(n_hops):
        start = hop * block_size
        chunk = audio[start:start + block_size]

        buffer = np.roll(buffer, -block_size)
        buffer[-block_size:] = chunk
        samples_received = min(buffer_len, samples_received + block_size)
        if samples_received < buffer_len:
            continue

        scores.append(_infer_score(buffer, model, scaler, feature_extractor, seq_len, n_mels, prep_params))

    return scores


# --- Worker process state (one ModelInference/Scaler per process; TFLite interpreters
# aren't fork/thread-shareable) ---
_worker_model = None
_worker_scaler = None
_worker_feature_extractor = None
_worker_seq_len = None
_worker_n_mels = None
_worker_prep_params = None
_worker_max_duration_sec = DEFAULT_MAX_DURATION_SEC
_worker_model_path = ""


def _init_worker(model_path, scaler_path, prep_params, max_duration_sec=DEFAULT_MAX_DURATION_SEC):
    global _worker_model, _worker_scaler, _worker_feature_extractor
    global _worker_seq_len, _worker_n_mels, _worker_prep_params, _worker_max_duration_sec, _worker_model_path

    _worker_model = ModelInference()
    _worker_model.load_model(model_path)
    input_shape = _worker_model.get_input_shape()
    _worker_seq_len = int(input_shape[1])
    _worker_n_mels = int(input_shape[2])
    _worker_model_path = model_path

    _worker_scaler = Scaler()
    mean, _var = _worker_scaler.load(scaler_path)
    if mean is None:
        raise RuntimeError(f"Failed to load scaler {scaler_path}: {_worker_scaler.last_error}")

    _worker_feature_extractor = FeatureExtractor()
    _worker_prep_params = prep_params
    _worker_max_duration_sec = max_duration_sec


def _score_one(path):
    scores = score_wav_file(
        path,
        _worker_model,
        _worker_scaler,
        _worker_feature_extractor,
        _worker_seq_len,
        _worker_n_mels,
        _worker_prep_params,
        max_duration_sec=_worker_max_duration_sec,
        model_path=_worker_model_path,
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

        key = _cache_key(path, args.model, args.scaler, prep_params, args.max_duration_sec)
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
                initargs=(args.model, args.scaler, prep_params, args.max_duration_sec),
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
            _init_worker(args.model, args.scaler, prep_params, args.max_duration_sec)
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
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=DEFAULT_MAX_DURATION_SEC,
        help="Score at most this many seconds per file, matching the live app's capture "
        "cap (main.py's Sliding Test Duration setting). Change only if that setting "
        "changed -- the two must agree or the fitted cutoff will not transfer.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

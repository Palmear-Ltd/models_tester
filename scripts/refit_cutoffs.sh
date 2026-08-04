#!/usr/bin/env bash
# Refit the EWMA-peak decision cutoff for every model, against a labeled corpus.
#
# Run this on the machine that holds the corpus. It scores the corpus once per model,
# fits a cutoff on a held-out train split, and writes the results to refit_out/ for
# review -- it deliberately does NOT overwrite models/*/decision_threshold.json, so a
# bad fit can never silently become the shipped verdict. The copy commands are printed
# at the end.
#
# Usage:
#   scripts/refit_cutoffs.sh /path/to/corpus [/path/to/another/corpus ...]
#
# Ground truth comes from a T/ or F/ path component under each corpus root, so pass the
# folder that CONTAINS T/ and F/, not T/ and F/ themselves.
#
# See docs/refit-decision-cutoff.md for the full runbook and the sanity checks.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <corpus-root> [more-corpus-roots...]" >&2
    exit 2
fi

PYTHON="${PYTHON:-.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-.score_cache}"
OUT_DIR="${OUT_DIR:-refit_out}"
WORKERS="${WORKERS:-8}"
# Must equal the app's Sliding Test Duration setting (Settings -> sliding_test_duration).
MAX_DURATION_SEC="${MAX_DURATION_SEC:-20}"

CORPUS_ARGS=()
for root in "$@"; do
    if [ ! -d "$root" ]; then
        echo "ERROR: corpus root not found: $root" >&2
        exit 1
    fi
    CORPUS_ARGS+=(--corpus "$root")
done

# model-dir-name:scaler-filename (9_1_1's scaler is spelled "scalar.json" in that drop)
MODELS=(
    "9_1_1:scalar.json"
    "9_1_2:scaler.json"
)

mkdir -p "$OUT_DIR"

for entry in "${MODELS[@]}"; do
    name="${entry%%:*}"
    scaler_file="${entry##*:}"
    model_path="models/$name/model.tflite"
    scaler_path="models/$name/$scaler_file"

    if [ ! -f "$model_path" ] || [ ! -f "$scaler_path" ]; then
        echo "SKIP $name (missing $model_path or $scaler_path)"
        continue
    fi

    echo
    echo "================ $name ================"
    mkdir -p "$OUT_DIR/$name"

    "$PYTHON" offline_score.py \
        "${CORPUS_ARGS[@]}" \
        --model "$model_path" \
        --scaler "$scaler_path" \
        --cache-dir "$CACHE_DIR" \
        --manifest-out "$OUT_DIR/$name/manifest.csv" \
        --max-duration-sec "$MAX_DURATION_SEC" \
        --workers "$WORKERS"

    "$PYTHON" evaluate_decision_rules.py \
        --manifest "$OUT_DIR/$name/manifest.csv" \
        --report-out "$OUT_DIR/$name/evaluation_report.csv" \
        --plots-dir "$OUT_DIR/$name/plots" \
        --threshold-out "$OUT_DIR/$name/decision_threshold.json" \
        | tee "$OUT_DIR/$name/evaluation.log"
done

echo
echo "=========================================================="
echo "Done. Review $OUT_DIR/*/evaluation.log before shipping."
echo
echo "To activate, copy each fitted cutoff next to its model:"
for entry in "${MODELS[@]}"; do
    name="${entry%%:*}"
    if [ -f "$OUT_DIR/$name/decision_threshold.json" ]; then
        echo "  cp $OUT_DIR/$name/decision_threshold.json models/$name/decision_threshold.json"
    fi
done

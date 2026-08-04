# Runbook: refit the infestation decision cutoff

Run this on the machine that holds the labeled corpus. Everything is already wired — you
should not need to edit any code.

For how the values are produced and which code to change when shipping a new one, see
[`decision-cutoff-maintenance.md`](decision-cutoff-maintenance.md).

## Why this needs doing

The decision cutoff is the single number a session verdict turns on: `EwmaPeakDecision`
smooths the per-window scores, tracks the running peak, and calls INFESTED if that peak
crosses the cutoff.

Two things are wrong with the current number:

1. **It's stale.** It was fitted in commit `61551ea`. Commit `77f1c3c` then added the
   buffer-fill hold-off, which changed what windows get scored — but nothing refit the
   cutoff. Dropping the leading near-silent windows means the EWMA now starts at a real
   score instead of ramping up from silence, which pushes session peaks *up*. The current
   cutoff is therefore slightly too permissive.
2. **`9_1_1` is using `9_1_2`'s number.** Only `models/9_1_2/` has a
   `decision_threshold.json`. `9_1_1` falls back to the shipped `DEFAULT_CUTOFF`, which is
   the 9_1_2 fit. A cutoff is specific to the model/scaler pair it was fitted against.

Since 2026-08-04 the app logs which of the two sources a cutoff came from, so you can see
this in the Status log at model load.

## Prerequisites

- The labeled corpus. Ground truth is resolved from a `T`/`F` **path component**, so point
  the script at the folder that *contains* `T/` and `F/`. The original fit used
  `test_data` plus `9_1_4/audio_data` (~6,500 sessions).
- A checkout of this branch with `requirements.txt` installed (needs tensorflow,
  librosa, soundfile, scikit-learn).
- Disk for the score cache: one small JSON per file per model.

## Steps

### 1. Preflight

```bash
python -c "import tensorflow, librosa, soundfile, sklearn; print('deps ok')"
python -m pytest tests/ -q          # expect all green
ls models/9_1_1/model.tflite models/9_1_2/model.tflite
```

Confirm the corpus resolves labels — this should print folders named `T` and `F`:

```bash
ls /path/to/corpus
```

### 2. Check one setting matches

The scorer caps each file at the app's capture duration. If you have changed **Settings →
Sliding Test Duration** away from 20 s, pass the same value via `MAX_DURATION_SEC` in step
4. If you have not touched it, skip this.

### 3. Smoke run (2 minutes)

Score 20 files first to confirm the plumbing before committing to the full corpus:

```bash
.venv/bin/python offline_score.py \
    --corpus /path/to/corpus \
    --model models/9_1_2/model.tflite --scaler models/9_1_2/scaler.json \
    --cache-dir .score_cache --manifest-out /tmp/smoke.csv \
    --limit 20
cut -d, -f2,8 /tmp/smoke.csv | head
```

**Expected:** labels are `T`/`F` (not blank), and `n_windows` is **36** for 20-second
recordings. Blank labels mean the corpus root is wrong. A number other than 36 means the
recordings aren't 20 s — note the actual length, it's fine, just tell me what you see.

### 4. Full refit, both models

```bash
scripts/refit_cutoffs.sh /path/to/corpus [/path/to/another/corpus ...]
```

Tunables via environment: `WORKERS` (default 8), `PYTHON`, `CACHE_DIR`, `OUT_DIR`,
`MAX_DURATION_SEC`. Scoring is cached per file, so a re-run after an interruption resumes
rather than starting over.

This writes, per model, into `refit_out/<model>/`: `manifest.csv`, `evaluation_report.csv`,
`plots/`, `evaluation.log`, and `decision_threshold.json`. **It does not touch
`models/`** — nothing goes live until you copy it in step 6.

### 5. Sanity-check the output

In each `refit_out/<model>/evaluation.log`:

- **`Loaded N labeled sessions`** — should be in the thousands. If it's tiny, most files
  failed label resolution; check the corpus root.
- **`Closest-edge disagreement`** — the ROC-optimal cutoff versus an independently fitted
  quantile band. The original fit agreed to 0.0004. The log flags anything above 0.01 with
  `<- LARGE, review before shipping`. If you see that, stop and send me the log.
- **The fitted cutoff itself** — the old value was `0.5697`. Something in roughly the
  0.4–0.7 range is expected. A wildly different number means something upstream is wrong.
- The summary table should still show `ewma_peak` at or near the top on
  `accuracy_conservative`. If some other rule now clearly wins, that's worth a conversation
  rather than a silent ship.

### 6. Activate

```bash
cp refit_out/9_1_1/decision_threshold.json models/9_1_1/decision_threshold.json
cp refit_out/9_1_2/decision_threshold.json models/9_1_2/decision_threshold.json
```

Then launch the app, pick each model, and confirm the Status log reads:

```
Decision cutoff: 0.XXXX (span=5.0)
  source: fitted for this model (decision_threshold.json)
```

If it still says `SHIPPED DEFAULT`, the file didn't land in the right folder.

### 7. Send back

Commit the two `decision_threshold.json` files, and send me `refit_out/*/evaluation.log`
plus `evaluation_report.csv`. I'll update `DEFAULT_CUTOFF` in `app/decision/threshold.py`
(the fallback for any future model with no fitted file) to match the 9_1_2 refit, and
record the new numbers in the docstring's provenance note.

## Notes

- **The score cache is safe across this change.** The cache key includes the windowing
  semantics (`SCORING_REV`), so entries written under the old zero-padded/uncapped rules
  can never be silently reused. An old `.score_cache` will simply be re-scored.
- **Both models share one cache dir** — the key includes model and scaler paths.
- `9_1_1`'s scaler file is spelled `scalar.json` (typo in that model drop). The script and
  the app both already handle it.

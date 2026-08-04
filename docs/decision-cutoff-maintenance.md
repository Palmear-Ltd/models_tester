# Decision cutoff: how the values are built and how to update the code

Maintainer-facing companion to [`refit-decision-cutoff.md`](refit-decision-cutoff.md).
That one is the operator runbook — *run these commands on the machine with the corpus*.
This one explains **where a cutoff value comes from** and **exactly what to change in the
code** once you have a new one.

---

## 1. What the cutoff is

A session produces one score per 0.5 s window. `EwmaPeakDecision`
(`app/decision/threshold.py:53`) reduces that sequence to a single number and compares it
to the cutoff:

```
smoothed[i] = α · score[i] + (1-α) · smoothed[i-1]      α = 2 / (span + 1)
peak        = max(smoothed)
verdict     = INFESTED if peak > cutoff else HEALTHY
```

Two parameters, always shipped together:

| | meaning | current default |
|---|---|---|
| `cutoff` | the peak value above which a session is called INFESTED | `0.5696557745704712` |
| `span` | EWMA smoothing window, in windows | `5.0` |

`span` sets how much smoothing happens before the peak is taken, so a cutoff is only valid
for the span it was fitted under. Never change one without refitting the other.

**`peak` is a running max.** This is the property that makes everything else in this
document matter: more windows in a session can only ever push the peak up, never down. So
a cutoff is valid only for sessions of roughly the length it was fitted on.

## 2. How a value is produced

```
corpus WAVs
   └─ offline_score.py                  → per-window score sequences (cached JSON)
        └─ manifest.csv                 → path, T/F label, month bucket, n_windows
             └─ evaluate_decision_rules.py
                  ├─ stratified 70/30 train/test split (seed=42)
                  ├─ fit on TRAIN only:  ROC-optimal (Youden J) cutoff on ewma_peak
                  ├─ cross-check:        fixed-n quantile band on the same statistic
                  └─ --threshold-out  → decision_threshold.json
```

Three properties of this pipeline are load-bearing:

- **`offline_score.py` must replicate the live loop exactly.** It is the definition of
  "what the app will see". If it scores windows the app never produces, the fitted cutoff
  is wrong in the app even though it looked right offline. See
  [`../offline_score.py`](../offline_score.py)'s module docstring for the full contract,
  and §5 below.
- **Fitting happens on the train split only**, so reported metrics come from sessions the
  cutoff never saw.
- **Two independent estimates must agree.** The ROC-optimal cutoff and the quantile band
  are fitted by unrelated methods; on the original fit they agreed to 0.0004. The
  evaluator prints the disagreement and flags anything above 0.01. Treat a large
  disagreement as "do not ship", not as "pick one".

## 3. Where values live at runtime

Resolution order in `main.py:load_resources`:

1. **`models/<model>/decision_threshold.json`** — the fitted value for that model/scaler
   pair. Preferred. Loading it requires no code change; drop the file in and restart.
2. **`DEFAULT_CUTOFF` / `DEFAULT_SPAN`** (`app/decision/threshold.py:24-25`) — the shipped
   fallback, used when a model directory has no JSON.

The app logs which one it used. `source: SHIPPED DEFAULT` means that model is running on
another model's number — see §4.2.

**A cutoff is specific to the model *and* scaler it was fitted against.** Scores from a
different model are a different distribution; the number does not transfer. Every model
that gets used for real testing wants its own JSON.

## 4. How to update the code

### 4.1 Shipping a newly fitted value (the common case)

The JSON files are data, not code — copy them in and commit:

```bash
cp refit_out/9_1_2/decision_threshold.json models/9_1_2/decision_threshold.json
```

Verify by loading the model in the app and reading the Status log:

```
Decision cutoff: 0.5697 (span=5.0)
  source: fitted for this model (decision_threshold.json)
```

Then update the fallback in `app/decision/threshold.py` to match the **primary** model's
refit (currently 9_1_2):

1. `DEFAULT_CUTOFF` (line 24) — paste the value at full precision, exactly as it appears
   in `evaluation.log`'s `ROC-optimal (Youden J) cutoff:` line. Do not round it; the JSON
   and the constant should be the same number.
2. `DEFAULT_SPAN` (line 25) — only if the span changed.
3. The module docstring (lines 1–13) — it is the provenance record. Update the corpus
   size, the split, the model it was fitted against, and the cross-check agreement figure.
   A future reader uses this to judge whether the number still applies to them.

Run the tests: `.venv/bin/python -m pytest tests/ -q`. `tests/decision/test_threshold.py`
covers the config round-trip and the accumulator.

### 4.2 Adding a cutoff for a model that has none

Nothing to code. Fit it (§2, or the runbook) and drop
`models/<name>/decision_threshold.json` in place. `_discover_models` and `load_resources`
find it by convention.

If the model directory uses a nonstandard scaler filename, check
`main.py:_discover_models` — it tries `scaler.json`, `scaler.npz`, `scalar.json` in that
order. `models/9_1_1/` relies on the third (a typo in that drop).

To include a new model in the refit script, add it to the `MODELS` array in
`scripts/refit_cutoffs.sh` as `<dir-name>:<scaler-filename>`.

### 4.3 Changing the span

`span` is currently defaulted in three places that must agree:

- `DEFAULT_SPAN` — `app/decision/threshold.py:25`
- `ewma_peak_score(scores, span=5.0)` — `app/decision/baselines.py:35`
- `EWMA_SPAN` — `evaluate_decision_rules.py` (imports `DEFAULT_SPAN`, so it follows)

Change the first two, then **refit** — a cutoff fitted at one span is meaningless at
another. The emitted JSON carries the span it was fitted under, so old JSONs stay
self-consistent.

## 5. What invalidates a cutoff

Any change to what windows get scored. The three loops below must stay in lockstep with
each other, and a change to any of them means the cutoff must be refitted:

- `main.py:mic_loop` — live capture
- `main.py:file_loop` — WAV replay
- `offline_score.py:score_wav_file` — corpus scoring

Guarded by `tests/test_input_path_parity.py` and `tests/test_offline_score_parity.py`.
The contract: 2.5 s rolling buffer at 0.5 s hop, buffer-fill hold-off, channel 0 only,
whole blocks only, capped at the live capture duration. A 20 s session yields **36** scored
windows on all three paths — a fast way to check you haven't drifted.

Also invalidating: preprocessing parameters (`DEFAULT_PREP_PARAMS` in `offline_score.py`
must mirror `main.py`'s Tk-var defaults), a new model or scaler, and the span.

When you change the windowing semantics, bump `SCORING_REV` in `offline_score.py`. It is
part of the score-cache key, so bumping it prevents a stale cache — scored under the old
rules — from being silently reused by the next refit.

## 6. Don't confuse these with the health thresholds

Same load-JSON-with-fallback idiom, unrelated purpose. The decision cutoff answers *is this
palm infested*; the ones below answer *is this signal trustworthy*, and are fitted from
hardware-condition data, not T/F infestation labels:

- `app/health/check_thresholds.json` — per-check thresholds
- `app/health/rootcause_session_config.json` — `DEFAULT_SESSION_CUTOFF`
  (`app/health/rootcause.py:131`), the SENSOR_LINK attribution floor
- `models/<model>/calibration.json` — health calibration profile

Refitting one has no bearing on the others.

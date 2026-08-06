# Manual test procedure: live mic + WAV replay

Runbook for the owner to manually exercise the tester GUI end-to-end — the one check
Claude can't do itself (no `$DISPLAY` / no piezo sensor in that environment). Covers the
2026-08 changes: per-model decision cutoffs, the new EWMA Decision Timeline panel, and
one-shot scoring.

## What changed and needs checking

| model | shipped cutoff? | expected on load |
|---|---|---|
| `9_1_1` | yes (new, today) | `Decision cutoff: 0.9979 (span=1.0)` / `source: fitted for this model` |
| `9_1_2` | yes (unchanged) | `Decision cutoff: 0.5697 (span=5.0)` / `source: fitted for this model` |
| `9_0_5` | **no** | `Decision cutoff: 0.5697 (span=5.0)` / `source: SHIPPED DEFAULT — not fitted for 9_0_5; verdicts are approximate` |
| `one_shot` | yes (new, today) | `Decision cutoff: 0.5279 (span=1.0)` / `source: fitted for this model` |

`9_0_5` showing `SHIPPED DEFAULT` is **expected**, not a bug — it's still on the
fallback by deliberate choice (see `docs/decision-cutoff-maintenance.md`).

## Prerequisites

- `source .venv/bin/activate` then `python launcher.py` from the repo root (needs the
  full-stack `.venv`: tensorflow, librosa, tkinter, sounddevice — `launcher.py` checks
  and installs anything missing).
- For the live-mic part: a piezo sensor connected. If none is available, skip Part 3 and
  do WAV-only.
- You do **not** need to relaunch the app between models in Part 1 — switching the model
  in Settings and clicking START TEST again reloads everything fresh each time.
- Sample WAV files are already in the repo — no external corpus needed:
  - `test_data/T/TP_9_1_1_20251224_101549.wav` (infested, 20.0s)
  - `test_data/F/TN_9_0_5_20251223_092425.wav` (healthy, 20.0s)

## Part 1 — Model load sanity check (repeat for all 4 models)

1. Launch the app. Input Source defaults to **Microphone** — if you don't have a sensor
   connected for this part, switch it to **Wav File** and **Browse** to
   `test_data/T/TP_9_1_1_20251224_101549.wav` first, so START TEST has something valid
   to run against regardless of model choice.
2. `⚙ Settings` → **Model & Scaler** tab → pick a model from the dropdown → **Close**.
3. Click **START TEST** — this is what actually triggers `load_resources()` (picking the
   model in the dropdown alone only stages the path; nothing loads until a test starts).
   The load-resources log lines below appear immediately, before any audio is captured.
   Click **STOP TEST** once you've checked them, then move to the next model.
4. In the Status log, check for, in order:
   - `Loaded Model: model.tflite`
   - `Model input shape: (np.int32(1), np.int32(98), np.int32(32), np.int32(1))` for
     `9_1_1`/`9_1_2`/`9_0_5`, or the same with `np.int32(784)` in the second slot for
     `one_shot` (the `np.int32(...)` wrapping is normal — that's just how this numpy
     version prints a shape tuple, not an error)
   - `Decision cutoff: ...` + `source: ...` — compare against the table above
5. For `one_shot` specifically, also confirm: **Inference Mode** in Settings →
   Acquisition & Output is forced to **Single Shot** (radio button greyed toward it /
   selected automatically) — a one-shot model should never run in sliding-window mode.

**Flag if:** the cutoff/span/source line doesn't match the table, the model fails to
load, or `one_shot` doesn't force single-shot mode.

## Part 2 — WAV replay test

1. Select model `9_1_2` (has the longest, most-validated track record) via Settings.
2. Input Source → **Wav File** → **Browse** → pick
   `test_data/T/TP_9_1_1_20251224_101549.wav` (labeled infested).
3. **START TEST**, let it run to completion (auto-stops at the configured duration).
4. Count the `Processed: ...` lines in the Status log for this session (there's no
   single summary line) — for a 20s file on a sliding-window model this should be
   **36** (not 40 — see `docs/decision-cutoff-maintenance.md` §5 if it isn't). For
   `one_shot`, expect exactly **1**. Easier to check precisely after the fact: see step 8
   below, `len(ewma_history)` in the saved JSON.
5. Check the **Dashboard**: `EWMA PEAK` value and `STATE` (HEALTHY/INFESTED).
6. Check the new **EWMA Decision Timeline** panel (bottom-ish strip in the plot area,
   above Health Timeline) — see "How to read the EWMA panel" below.
7. Repeat with `test_data/F/TN_9_0_5_20251223_092425.wav` (labeled healthy) and confirm
   the verdict flips the other way.
8. If **Save results and audio** is checked (Settings → Acquisition & Output), open the
   saved `.json` next to the WAV and confirm an `"ewma_history"` key is present — a list
   of `[time, smoothed, peak]` triples, one per scored window (36 entries for a
   sliding-window model, 1 for `one_shot`).

## Part 3 — Live mic test

1. Same as Part 2, but Input Source → **Microphone**, pick the right device (use
   **⟳ Refresh** if it's not listed), sensor connected and making realistic contact.
2. **START TEST**, let a full session run (auto-stops per **Sliding Test Duration** in
   Settings, default 20s), then check the same things as Part 2 steps 4-8.
3. Also check **Signal Health Detail** (the health-check tree) is populating — this is
   the unrelated-but-adjacent audio-health subsystem; it should show PASS/WARNING/FAIL
   rows, not stay blank or all `NOT_EXECUTED`.
4. Try a deliberately bad recording (e.g. lightly tap the sensor, or briefly detach it)
   and confirm the health indicator reacts (this exercises the older, unrelated
   sensor-link work — a quick regression check, not new for today).

## How to read the EWMA Decision Timeline panel

Three lines, all on a 0-1 y-axis against session time on the x-axis:

- **Blue solid — EWMA.** The smoothed score at each window. This is what "how the score
  is evolving" looks like moment to moment.
- **Purple dashed — Peak.** The running max of the blue line so far. This is what the
  final verdict is actually based on (`EwmaPeakDecision.peak`) — it can only go up.
- **Red dotted — Cutoff.** The model's active decision threshold. Peak crossing this
  line = INFESTED; staying under it for the whole session = HEALTHY.

**Span shows up visually.** `9_1_1` and `one_shot` were both fit at `span=1.0` (no
smoothing) — their blue line should look jagged, tracking the raw per-window score
almost exactly, and for `one_shot` there's only a single point (n=1, nothing to draw a
line through). `9_1_2` is at `span=5.0` — its blue line should look visibly smoother.
Both are correct; a smoother or jumpier line is not itself a bug, it's the fitted span
doing its job. What *would* be a bug: the purple peak line ever decreasing, or the blue
line exceeding the purple line (peak must always be ≥ current smoothed value).

## How to analyze a session overall

1. **Does the verdict match the label?** A `test_data/T/*` (`TP_` prefix) file should
   land on INFESTED; a `test_data/F/*` (`TN_` prefix) file should land on HEALTHY. A
   mismatch on these specific files would be a real problem — they're the model's own
   training-adjacent sanity set.
2. **Does the peak clear the cutoff by a comfortable margin, or barely graze it?** A
   verdict that only just crosses (or just misses) the line is expected to be less
   reliable than one with daylight between peak and cutoff — that's inherent to any
   cutoff-based decision, not something to "fix."
3. **Cross-check n_windows.** 36 for a 20s sliding-window session, 1 for one-shot. Wrong
   counts mean the windowing pipeline has drifted (see
   `docs/decision-cutoff-maintenance.md` §5) — a real, actionable bug if seen.
4. **Cross-check the saved JSON's `ewma_history`** against what the panel showed live —
   they should match exactly (same source, `self.ewma_history`, just serialized).
5. **For `9_0_5`:** treat any verdict as approximate per the `SHIPPED DEFAULT` warning —
   don't read too much into a borderline call on this model specifically until it has
   its own fitted cutoff.

## Known weak spots (not new bugs if you see them)

If you test with recordings beyond the two `test_data/` sanity files, keep these
already-investigated findings in mind so they don't read as new problems:

- **`9_0_5` is the weakest of the four models generally** — lowest accuracy in its refit,
  and it missed a known-infested sanity file even when tried against its own best fitted
  cutoff (not just the borrowed fallback). Don't trust a borderline `9_0_5` call much.
- **Recordings from Jan–Apr 2021 are a known hard spot for `9_1_1`, `9_1_2`, and
  `one_shot`** (elevated false-negative *and* false-positive rates in that window,
  fading to normal by mid-2021) — this also shows up in the old production rule, so it
  predates all of today's changes and isn't specific to any model. We checked whether
  this is a recurring seasonal (temperature/larval-activity) effect by comparing the
  same calendar months across other years — it isn't; only that specific historical
  window is affected. If you're testing with vintage 2021 field recordings specifically,
  expect worse accuracy than usual across the board.
- **A verdict that barely crosses (or barely misses) the cutoff is not a red flag on its
  own** — several real recordings score within ~0.02 of the line in either direction.
  Only a *wrong* verdict (not a close one) on a confidently-labeled file is worth
  investigating.

## What would indicate an actual problem (vs. expected behavior)

- Any `Error` line in the Status log during model load, inference, or health checks.
- Wrong `source:` line for a given model (cross-check against the table above).
- Wrong window count for a 20s test (not 36 for sliding, not 1 for one-shot).
- EWMA panel not updating during a live/replay session, or peak line ever decreasing.
- Saved results JSON missing `ewma_history`, or its length not matching the window
  count.
- App crash, freeze, or the Settings dialog failing to reflect the just-picked model's
  path/scaler fields.

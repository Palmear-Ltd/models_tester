# RMS-based verdict confidence — design

## Problem

`main.py:run_inference` already computes RMS energy on every 2.5s buffer it feeds the
model (`main.py:1201`, `rms = np.sqrt(np.mean(audio_data**2))`) and tracks it across the
session (`self.energy_history`, `main.py:111`). Today that number drives only the live
"Current Energy" bar (`main.py:306`) — it is discarded once the window is scored, never
reaches the model's features (which are floor-normalized per window — see
`app/audio/features.py`'s 5th-percentile-floor step — so absolute loudness is not
implicitly available to the CNN either), and plays no role in the session verdict or in
how much a tester should trust it.

A corpus-wide investigation this session (6,097 files, `9_1_4/audio_data/wav/{T,F}`,
2021–2026) found RMS is a weak direct predictor of ground truth (ranked-AUC 0.33–0.43,
close to uninformative) but a much stronger predictor of **whether the model's verdict on
this particular session is likely correct** — conditioned on which way the model called
it:

| pool (conditioned on the model's own verdict) | question | AUC |
|---|---|---|
| predicted INFESTED (TP vs FP) | does low RMS → more likely correct (TP)? | 0.72 (mean) / 0.76 (peak) |
| predicted HEALTHY (TN vs FN) | does high RMS → more likely correct (TN)? | 0.71 (mean) / 0.74 (peak) |

(Figures use raw/broadband RMS — the same quantity `main.py` already computes — on the
`9_1_4/audio_data/wav/{T,F}` corpus with 2021 excluded; see Evidence below.) This is the
first RMS signal found this session strong enough to act on.

## Why this matters (motivating use case)

A tester gets a bare `INFESTED`/`HEALTHY` call today (`main.py:calculate_diagnosis`,
`main.py:629-651`) with no sense of whether the *specific recording conditions* of this
session make the call more or less trustworthy. The existing `app/health/rootcause.py`
SENSOR_LINK mechanism (the "Validate Acquisition" button, the live "Likely cause" label)
answers a related but different question — "is there a sensor/cable fault, independent of
what the model said" — via an omnidirectional Mahalanobis distance from a calibration
profile centroid (`app/health/anomaly.py`), which treats "too loud" and "too quiet"
symmetrically and requires a manually-loaded calibration profile that is off by default
(`self.calibration_profile = None` at startup — see project memory
`project-calibration-json-semantics`).

The RMS-verdict-confidence signal this spec proposes is different in kind: it is
**directional** (loud specifically undermines an INFESTED call; quiet specifically
undermines a HEALTHY call) and needs no calibration profile — it only needs the session's
own RMS trace, which is already computed unconditionally today.

## Evidence

Source: this session's investigation, `9_1_4/audio_data/wav/{T,F}` corpus (the
authoritative multi-year source per `reference-9-1-4-external-corpus`), scored with the
shipped `models/9_1_2` model + cutoff (`decision_threshold.json`: cutoff=0.5696557745704712,
span=5.0). 2021 excluded per owner direction (that year has deliberately-introduced noise,
and was 39% of T files vs. 15% of F files — an imbalance that could bias either
direction). n=4,775 files (T=1,028, F=3,747) after exclusion; excluding 2021 did not
weaken any of the findings below — if anything it sharpened them.

**Raw RMS is a weak direct predictor of ground truth** (T vs. F): rank-AUC 0.35 (mean) /
0.30 (peak), Cohen's d ≈ −0.37 / −0.41. Real, but too weak and too confounded (see below)
to use as a standalone classifier feature.

**RMS conditioned on the model's own verdict is a much stronger signal of session
trustworthiness** — this is the part being proposed for use:

| bucket (shipped cutoff) | n | mean-RMS vs. TN | peak-RMS vs. TN |
|---|---|---|---|
| TN (correct HEALTHY) | 2,456 | 1.00× | 1.00× |
| **FP** (wrong INFESTED call) | 1,291 | **1.93×** | **1.81×** |
| TP (correct INFESTED) | 931 | 0.58× | 0.57× |
| **FN** (wrong HEALTHY call) | 97 | **0.44×** | **0.39×** |

False positives run hot (elevated RMS — consistent with the previously root-caused
SENSOR_LINK contact-noise pattern, see `project-fp-field-session-sensor-link`); false
negatives run quiet (consistent with insufficient signal energy to register the actual
click pattern). Held-out logistic fits (log(RMS) → P(verdict correct), 70/30 split, seed
42) confirm this isn't a `pred`-leakage artifact: AUC 0.67 (INFESTED-pool) / 0.77
(HEALTHY-pool) on the held-out test fold.

**Robustness checks already run:**
- Bandpass-filtering RMS to the model's 500–8000Hz analysis band first (matching
  `app/audio/processor.py:AudioProcessor.bandpass_filter`, off by default in the app)
  weakens the raw T-vs-F ground-truth effect substantially (rank-AUC moves from
  0.35→0.38/0.30→0.38, closer to uninformative) — confirming that effect is partly a
  broadband recording-condition artifact, not (only) bioacoustic. But it leaves the
  verdict-conditional FP/FN pattern **essentially unchanged** (FP still ~1.9–2.1× TN, FN
  still ~0.3–0.4× TN) — this is the evidence this spec's proposal is built on, and it
  survives band-limiting.
- Excluding 2021 (deliberately-noised year) did not weaken the verdict-conditional
  pattern either (FP/TN ratio 1.86×→1.93×; FN/TN ratio 0.45×→0.44×, bandpassed FN/TN
  0.40×→0.29×, i.e. stronger).

**Known weak spot:** the FN branch has only 97 examples in the whole multi-year corpus.
The held-out AUC (0.77) is real but the fitted curve's coefficients will have wide
uncertainty — flagged as a first-class risk below, not glossed over.

## Design

### 1. RMS aggregate: session mean, raw/broadband (not bandpassed)

Use the session's **mean RMS** across scored windows — the same quantity already
accumulated in `self.energy_history` (`main.py:111`), no new signal acquisition. Peak-RMS
scored comparably or slightly better in places (see Evidence) and is worth comparing
during fitting, but mean is the simpler, already-visible number and the fit script should
just pick empirically whichever generalizes better on the held-out split — this is not a
hardcoded pre-decision.

Ship on **raw/broadband** RMS, matching exactly what `main.py:run_inference` already
computes (`use_filter` stays independent — most users leave the bandpass filter off, and
main.py's RMS calc happens before any filtering regardless of that setting). Bandpassed
RMS was evaluated as an alternative (see Evidence) and does not clearly outperform raw for
this specific verdict-conditional use, so there's no case for adding a second filtering
pass to a value that's already computed for free.

### 2. Two small fitted curves, one per verdict branch

Mirrors `app/decision/threshold.py`'s `ThresholdConfig`/`default_config()` pattern
exactly — a config dataclass with a few data-fit numbers, not a bigger model:

```
P(correct | INFESTED, mean_rms) = sigmoid(a1 * log(mean_rms) + b1)   # a1 < 0
P(correct | HEALTHY, mean_rms)  = sigmoid(a2 * log(mean_rms) + b2)   # a2 > 0
```

Fit via the same train/test split machinery already in `evaluate_decision_rules.py`
(`stratified_split`, seed=42) so the fit uses an identical, already-established
methodology and doesn't leak into whatever this or a future cutoff refit evaluates
against.

### 3. Where it plugs in

**Not** `app/health/` — `HealthAnalysisPipeline.analyze()` runs per-window
(`app/health/pipeline.py:27`), before the session verdict exists, and per
`app/health/`'s hard portability convention never imports `app/decision/` or anything
verdict-aware. This is a **session-level, post-verdict** concept, so it belongs beside
`app/decision/threshold.py`, applied in `main.py:calculate_diagnosis` right after
`EwmaPeakDecision` resolves `state`/`peak` (`main.py:642-651`), using
`self.energy_history`'s mean.

### 4. Additive only — never touches the verdict

Same hard convention `app/health/anomaly.py`'s result already follows for
`HealthReport.confidence` ("anomaly sets confidence only, never state" —
`app/health/fusion.py:95-103`): this is a second, independent confidence value
surfaced *alongside* `state`, never allowed to flip or suppress it. `state`/
`predicted_infested` stay exactly as `EwmaPeakDecision` computes them.

### 5. Report coarse trust tiers, not a bare float

Given the FN branch's data-thinness (n=97), a precise-looking decimal overstates what's
known. Bucket into **low / medium / high** trust via fixed cutoffs on the fitted
probability, chosen during fitting/validation (not guessed blind), and surface the tier —
e.g. `"INFESTED (EWMA peak: 0.71) — verdict confidence: low (energy unusually high for
this call)"` / `"HEALTHY (EWMA peak: 0.31) — verdict confidence: low (signal unusually
quiet — verify sensor coupling)"`.

### 6. RMS needs to become a captured, reproducible pipeline input

Today RMS is not part of `offline_score.py`'s cache or `manifest.csv` at all — this
session's investigation used a standalone companion script, `scripts/rms_scan.py`
(already added to the repo, mirrors `offline_score.py`'s exact windowing/hold-off/cap so
window *i* lines up 1:1 with score cache window *i*), with its own cache dir
(`.rms_cache/`) keyed independently of the model. That script should become the
permanent, documented way to backfill RMS onto an existing scored manifest — **not**
folded into `offline_score.py`'s own cache (that would require bumping `SCORING_REV` and
re-running TFLite inference over the whole corpus just to add a cheap RMS column, which is
wasteful; RMS computation is orders of magnitude cheaper than model inference and doesn't
need to share a cache lifecycle with it).

## Non-goals

- Not touching the CNN or its training (training doesn't happen in this repo — see
  `9_1_4/train.ipynb`). This is a post-hoc session-level trust signal, not a model change.
- Not replacing or duplicating `app/health/rootcause.py`'s SENSOR_LINK mechanism or
  `app/health/anomaly.py`'s Mahalanobis confidence — those stay as-is; this is a third,
  complementary, verdict-aware signal that doesn't require a loaded calibration profile.
- Not adopting the raw T-vs-F ground-truth RMS effect as a feature — it's real but weak
  and partly confounded (see Evidence); only the much stronger verdict-conditional signal
  is being proposed for use.
- Not adding bandpass filtering to `main.py`'s live RMS computation.

## Risks / open questions

- **FN branch is data-thin (n=97 corpus-wide).** The fitted curve's held-out AUC (0.77) is
  real, but implementation must not present its output with false precision (hence coarse
  tiers, not a raw float) and should log/flag when a session's confidence relies on the
  thin branch, so it's easy to revisit once more field FN data accumulates. If the fitted
  logistic proves unstable (wide bootstrap CI on `a2`/`b2`) during implementation, fall
  back to a simpler threshold-band rule for that branch specifically rather than force a
  logistic fit onto too little data — decide during implementation with the actual fit
  diagnostics in hand, not preemptively here.
- **Peak vs. mean RMS** — pick empirically during fitting (see Design §1), not here.
- **Interaction with a loaded calibration profile.** When both anomaly-confidence
  (`app/health/`) and this verdict-confidence (`app/decision/`) are active, they are
  computed independently and both displayed — no fusion between them is proposed here;
  that's a follow-up if it proves confusing in practice.

## Acceptance / validation

Replay the held-out test split (or the local `test_data/{T,F}` corpus, thin but useful as
an existence check) through the fitted curves:
- Within the predicted-INFESTED pool: mean confidence should be visibly higher for TP than
  FP.
- Within the predicted-HEALTHY pool: mean confidence should be visibly higher for TN than
  FN (informative only given n=97 corpus-wide — no strong statistical claim, direction
  check).
- `state`/`predicted_infested` must be byte-identical to the pre-change behavior on every
  replayed file — this feature must never change a verdict.

Plus synthetic-signal unit tests per project TDD convention: synthetic sessions at
controlled RMS levels straddling the fitted branch cutoffs, for both verdict branches;
missing-config-file fallback; JSON round-trip.

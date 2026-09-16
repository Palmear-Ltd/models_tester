# 9_1_5 (tester build, 2026-09-16 — second cut)

Retrained for the thin piezo→preamp cable. Built in `../../training/9_1_5/`
(read its `README.md` for the data, the recipe and every number below); this is
`runs/run4all_s43/release/` — the run-3 recipe (app-parity features, 2025–2026 +
the Aug/Sep 2026 field sessions only, ≤ 28 chunks per file, SpecAugment, balanced
class weights, early stop on file-level val AUC), seed 43, **now trained on the
2026-09-15 standard-vs-thin twin session as well** (27 infested + 25 healthy
trees, both cables).

Why the second cut: the first cut (`runs/run3_s43`, shipped 2026-09-13) missed
8 of 27 infested trees on the thin cable in the 2026-09-15 field test while 9_1_2
caught all 27 on the very same recordings — it had learned "thin-cable timbre ⇒
healthy" from a training set with 142 thin-cable healthy files and 57 thin-cable
positives from a single tree. The recipe was re-validated with a tree-level
hold-out of the new session first (`run4_s43`: 1/11 held-out thin-cable infested
trees missed under the phone's count rule vs 3/11 for the first cut, 0/10 false
positives), then retrained with every tree in train for this artifact.

Files: `model.tflite` (input `[1,98,32,1]`, sigmoid), `scaler.json` (refit for
this model — do not reuse 9_1_2's or the first cut's), `decision_threshold.json`
(EWMA span 8, **cutoff 0.50**, carried over from the hold-out pick; the val-fitted
Youden cutoff was 0.709), `model_params.json` (app count rule, `labelScore` 0.17
kept from the first cut — the val fit without thin-cable positives said 0.47,
which would trade false negatives for false positives, the opposite of the goal).
No `rms_confidence.json` / `calibration.json` — the tester falls back to the
shipped defaults and says so in the log.

Session-level on the 9_1_5 test split (1,515 recordings, unchanged since the
13 Sep build; EWMA-peak rule at each model's cutoff):

| | FNR | FPR | AUC | 2026 FNR / FPR | 2026 AUC |
|---|---|---|---|---|---|
| 9_1_2 | 0.115 | 0.336 | 0.865 | 0.000 / 0.328 | 0.966 |
| 9_1_5 first cut | 0.158 | 0.188 | 0.902 | 0.086 / 0.136 | 0.949 |
| **9_1_5 this cut** | 0.145 | 0.209 | 0.899 | 0.049 / 0.173 | 0.963 |

New-data sessions 0/60 missed (first cut 0/60 tester rule, 2/60 app rule → now
1/60); thin cable 10/10 detected, 3/98 FP; manual_test (the 10.08 misses) 9/17,
unchanged. On the 2026-09-15 session itself (training data now, sanity only):
0 FN / 0 FP on both cables under the phone rule, the weakest infested thin-cable
recording at 26/40 hops. The field test that matters next is a thin-cable
session on trees this model has not seen.

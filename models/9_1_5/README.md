# 9_1_5 (candidate, 2026-09-13)

Retrained for the thin piezo→preamp cable. Built in `../../training/9_1_5/`
(read its `README.md` for the data, the recipe and every number below); this is
`runs/run3_s43/release/` — the run-3 recipe (app-parity features, 2025–2026 + the
Aug/Sep 2026 cable/housing sessions only, ≤ 28 chunks per file, SpecAugment,
balanced class weights, early stop on file-level val AUC), seed 43, chosen on val.

Files: `model.tflite` (input `[1,98,32,1]`, sigmoid), `scaler.json` (refit for
this model — do not reuse 9_1_2's), `decision_threshold.json` (EWMA span 8,
**cutoff 0.50 chosen by the owner**; the val-fitted Youden cutoff was 0.742),
`model_params.json` (app count rule; `labelScore` refit to 0.17 on val,
`sumInfThr`/`sumSusThr` unchanged). No `rms_confidence.json` / `calibration.json`
yet — the tester falls back to the shipped defaults and says so in the log.

Session-level on the 9_1_5 test split (1,515 recordings, EWMA-peak rule, cutoff 0.50)
vs 9_1_2 at its shipped cutoff:

| | FNR | FPR | bal. acc | AUC |
|---|---|---|---|---|
| 9_1_2 | 0.115 | 0.336 | 0.774 | 0.865 |
| 9_1_5 | 0.158 | 0.188 | 0.827 | 0.902 |

Thin cable (10 infested / 98 healthy recordings): 10/10 detected, 2/98 false
positives, verdict agrees with the standard-cable twin on 20/20 paired recordings.
Known soft spot: the healthy tree recorded on 2026-09-02 with the *standard* cable
gives 15/30 false positives at this cutoff (9_1_2: 2/30). The only thin-cable
infested recordings so far come from one tree on one morning — the field test to
run next is a second infested tree and a healthy tree with the thin cable.

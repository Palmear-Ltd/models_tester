one_shot_2 -- FIELD-TEST CANDIDATE, NOT YET APPROVED FOR RELEASE (2026-10-20)

One-shot model: one inference on the whole first 20 s of a recording.
Inside model.tflite: three all-years models averaged (training/one_shot_1, ens_pool_m_all).
Input [1, 784, 32, 1]; same features as the app's one-shot path. Cutoff 0.479924 (span 1.0).

Files (same format as models/one_shot): model.tflite, scaler.json, decision_threshold.json.
Checksums: SHA256SUMS.

Field notes:
- Record the full 20 s; shorter recordings are padded with silence.
- Keep the app's file naming (TP/TN/FP/FN prefix): the tree's true state is read from it.
- Upload every recording, whatever the verdict, as in drive-upload-instructions.md.
- Never re-record or drop a tree because of what the model said: the comparison needs every tree.

Release decision: training/one_shot_1/docs/superpowers/specs/2026-10-20-follow-up-ensemble.md

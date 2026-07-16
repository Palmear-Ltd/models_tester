# Calibration profiles

Health-monitoring baseline profiles (`app/health/calibration.py`), consumed via
Settings → Model & Scaler → Calibration Profile. These characterize the
piezo sensor/cable/recording-chain's normal acoustic and electrical range —
they are **not** tied to a specific `.tflite` model version (the CNN never
reads this data; only `app/health/`'s signal-quality checks and Mahalanobis
anomaly detector do).

## multiyear_healthy_v1.json

Generated 2026-07-16 from 450 `TN_`-prefixed (correctly-classified-healthy)
recordings sampled from `/home/bashar/workspace/palmear/9_1_4`, spanning
2021-2026 and five hardware variants (standard cable, thin_cable,
thick_cable, an Android recording device, and the "china needle" sensor).
`FP_`-prefixed files in the same source folders were deliberately excluded
even though their ground truth is also "healthy" — they were misclassified
by the model, often due to sensor-link artifacts, and including them would
have baked that noise into the "normal" baseline.

45,872 windows, 17 checks characterized, 28-feature covariance model.

The raw WAV corpus used to generate this was deleted after generation (2.8 GB,
reproducible via the sampling logic below — it isn't checked into this repo).
To regenerate or rebuild with different sampling:

```bash
.venv/bin/python calibrate.py \
  --input <folder-of-known-healthy-TN-only-recordings> \
  --output calibration_profiles/<name>.json \
  --profile-id <name> \
  --sensor-info "<hardware description>"
```

import json

import numpy as np

from app.audio.scaler import Scaler


def test_load_valid_json_returns_mean_var(tmp_path):
    p = tmp_path / "scaler.json"
    p.write_text(json.dumps({"mean": [0.0] * 32, "var": [1.0] * 32}))
    scaler = Scaler()
    mean, var = scaler.load(str(p))
    assert mean is not None and var is not None
    assert mean.shape == (32,)
    assert scaler.last_error is None


def test_invalid_npz_returns_none_and_records_reason(tmp_path):
    # An HTML page saved with a .npz extension (the real-world failure mode).
    p = tmp_path / "scaler.npz"
    p.write_bytes(b"\n\n<!DOCTYPE html>\n<html>not a numpy file</html>")
    scaler = Scaler()
    mean, var = scaler.load(str(p))
    assert mean is None and var is None
    assert scaler.last_error
    assert "scaler.npz" in scaler.last_error


def test_npz_missing_keys_records_reason(tmp_path):
    p = tmp_path / "wrongkeys.npz"
    np.savez(str(p), foo=np.zeros(32))
    scaler = Scaler()
    mean, var = scaler.load(str(p))
    assert mean is None and var is None
    assert scaler.last_error
    assert "mean" in scaler.last_error


def test_json_missing_mean_records_reason(tmp_path):
    p = tmp_path / "nomean.json"
    p.write_text(json.dumps({"var": [1.0] * 32}))
    scaler = Scaler()
    mean, var = scaler.load(str(p))
    assert mean is None and var is None
    assert scaler.last_error
    assert "mean" in scaler.last_error

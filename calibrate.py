"""CLI: generate a CalibrationProfile JSON from healthy WAV recordings.

Usage:
  ./.venv/bin/python calibrate.py --input <folder-or-wav> --output profile.json \
      --profile-id piezo_v1 [--sensor-info "piezo, housing A"]

WAV loading (soundfile/librosa) lives here so app/health stays NumPy-only.
"""
import argparse
import glob
import os

import librosa
import numpy as np
import soundfile as sf

from app.health.calibration import generate_profile, save_profile

TARGET_SR = 44100


def _load_wav(path):
    data, fs = sf.read(path, always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    if fs != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=fs, target_sr=TARGET_SR).astype(np.float32)
    return mono


def _gather_wavs(input_path):
    if os.path.isdir(input_path):
        # Case-insensitive .wav match so .WAV files (mixed-source recordings) aren't skipped.
        all_files = glob.glob(os.path.join(input_path, "**", "*"), recursive=True)
        return sorted(p for p in all_files if p.lower().endswith(".wav"))
    return [input_path]


def run(input_path, output_path, profile_id, sensor_info=""):
    paths = _gather_wavs(input_path)
    if not paths:
        raise SystemExit(f"No WAV files found at {input_path}")
    signals = [_load_wav(p) for p in paths]
    profile = generate_profile(
        signals, TARGET_SR, profile_id=profile_id, sensor_info=sensor_info
    )
    save_profile(profile, output_path)
    print(
        f"Wrote {output_path}: {profile.window_count} windows from {len(paths)} "
        f"file(s), {len(profile.statistics)} checks characterized."
    )
    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Generate a calibration profile from healthy WAV recordings."
    )
    parser.add_argument("--input", required=True, help="WAV file or folder (recursive)")
    parser.add_argument("--output", required=True, help="Output profile JSON path")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--sensor-info", default="")
    args = parser.parse_args()
    run(args.input, args.output, args.profile_id, args.sensor_info)


if __name__ == "__main__":
    main()

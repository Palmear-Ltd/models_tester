"""Filename/path metadata parsing for labeled WAV corpora.

Pure stdlib (re + datetime) — no disk I/O, independently unit-testable. Handles the mixed
naming conventions observed across the labeled corpora (e.g. `TP_080621_0757_....wav` /
`FN_9_1_2_20260106_104218.wav`).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

_YEAR_MIN = 2015
_YEAR_MAX = 2035

_YYYYMMDD_RE = re.compile(r"(?:^|_)(\d{4})(\d{2})(\d{2})(?:_|\.)")
_DDMMYY_RE = re.compile(r"(?:^|_)(\d{2})(\d{2})(\d{2})(?:_|\.)")


def _valid_date(year: int, month: int, day: int) -> Optional[date]:
    if not (1 <= month <= 12):
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_recording_date(filename: str) -> Optional[date]:
    """Best-effort recording date extraction from a WAV filename.

    Tries every 8-digit YYYYMMDD group first (unambiguous), then every 6-digit DDMMYY
    group, returning the first one that validates as a real date in a plausible year
    range. Returns None rather than guessing when nothing plausible is found.
    """
    for m in _YYYYMMDD_RE.finditer(filename):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _YEAR_MIN <= year <= _YEAR_MAX:
            d = _valid_date(year, month, day)
            if d is not None:
                return d

    for m in _DDMMYY_RE.finditer(filename):
        day, month, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 2000 + yy
        if _YEAR_MIN <= year <= _YEAR_MAX:
            d = _valid_date(year, month, day)
            if d is not None:
                return d

    return None


def season_bucket(d: date) -> str:
    """Calendar-month bucket, e.g. '2021-06'. Avoids assuming a hemisphere-specific season."""
    return f"{d.year:04d}-{d.month:02d}"


def resolve_label(path: str) -> Optional[str]:
    """Ground-truth label ('T'/'F') from a path's directory components, or None if absent.

    Matches both `test_data/{T,F}/...` and every `.../<category>/{T,F}/...` layout in the
    external corpus.
    """
    for part in re.split(r"[/\\]", path):
        if part in ("T", "F"):
            return part
    return None


def is_ambiguous_prefix(filename: str) -> bool:
    """True if the filename carries an ambiguous/unlabeled 'X_' prefix that should be
    excluded from evaluation rather than silently trusted."""
    base = re.split(r"[/\\]", filename)[-1]
    return base.startswith("X_")

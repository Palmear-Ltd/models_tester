from datetime import date

from app.decision.manifest import (
    is_ambiguous_prefix,
    parse_recording_date,
    resolve_label,
    season_bucket,
)


def test_parse_recording_date_ddmmyy_convention():
    assert parse_recording_date("TP_080621_0757_rtOt55vIDGyJ7CQnAw2n.wav") == date(2021, 6, 8)
    assert parse_recording_date("FP_220621_0823_LNqS3okD34b8b7kGBNPM.wav") == date(2021, 6, 22)


def test_parse_recording_date_yyyymmdd_convention():
    assert parse_recording_date("TP_9_1_1_20251224_101726.wav") == date(2025, 12, 24)
    assert parse_recording_date("TN_9_0_5_20251223_092523.wav") == date(2025, 12, 23)
    assert parse_recording_date("FP_9_1_1_20260309_095136.wav") == date(2026, 3, 9)


def test_parse_recording_date_returns_none_for_garbage():
    assert parse_recording_date("not_a_date_file.wav") is None
    assert parse_recording_date("recording_final_v2.wav") is None


def test_parse_recording_date_rejects_implausible_year():
    # A 6-digit group that parses as a date far outside any plausible recording year.
    assert parse_recording_date("clip_999999_take2.wav") is None


def test_season_bucket_format():
    assert season_bucket(date(2021, 6, 8)) == "2021-06"
    assert season_bucket(date(2026, 1, 1)) == "2026-01"


def test_resolve_label_in_repo_layout():
    assert resolve_label("test_data/T/TP_9_1_1_20251224_101726.wav") == "T"
    assert resolve_label("test_data/F/TN_9_0_5_20251223_092523.wav") == "F"


def test_resolve_label_external_corpus_layout():
    assert resolve_label("/home/x/9_1_4/audio_data/wav/T/TP_080621_0757_abc.wav") == "T"
    assert resolve_label("/home/x/9_1_4/audio_data/thick_cable/F/clip.wav") == "F"


def test_resolve_label_none_when_absent():
    assert resolve_label("/home/x/9_1_4/audio_data/apple_adapter/clip.wav") is None


def test_is_ambiguous_prefix():
    assert is_ambiguous_prefix("X_unlabeled_clip.wav") is True
    assert is_ambiguous_prefix("test_data/F/X_unlabeled_clip.wav") is True
    assert is_ambiguous_prefix("TP_080621_0757_abc.wav") is False

"""Case loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.corpus.builder import build
from bench.corpus.manifest import Bucket, Manifest, ManifestEntry
from bench.runner.case import (
    DEFAULT_PRESETS,
    PRESET_QUALITY,
    CorpusFileMissing,
    load_cases,
)


def _manifest() -> Manifest:
    return Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="photo_a",
                bucket=Bucket.SMALL,
                content_kind="photo_noise",
                seed=1,
                width=192,
                height=144,
                output_formats=["png"],
                tags=["photo"],
            ),
            ManifestEntry(
                name="photo_b",
                bucket=Bucket.SMALL,
                content_kind="photo_noise",
                seed=2,
                width=192,
                height=144,
                output_formats=["png"],
                tags=["photo", "noise"],
            ),
        ],
    )


def test_preset_quality_mapping_is_complete():
    assert set(PRESET_QUALITY) == set(DEFAULT_PRESETS)
    assert PRESET_QUALITY["high"] < PRESET_QUALITY["medium"] < PRESET_QUALITY["low"]


def test_load_cases_emits_one_per_format_preset(tmp_path: Path):
    m = _manifest()
    build(m, tmp_path)

    cases = load_cases(m, tmp_path)
    # 2 entries × 1 format × 3 presets
    assert len(cases) == 6


def test_load_cases_case_id_format(tmp_path: Path):
    m = _manifest()
    build(m, tmp_path)
    cases = load_cases(m, tmp_path)
    case = next(c for c in cases if c.name == "photo_a" and c.preset == "high")
    assert case.case_id == "photo_a.png@high"


def test_load_cases_quality_matches_preset(tmp_path: Path):
    m = _manifest()
    build(m, tmp_path)
    cases = load_cases(m, tmp_path)
    qualities = {c.preset: c.quality for c in cases}
    assert qualities == PRESET_QUALITY


def test_load_cases_format_filter(tmp_path: Path):
    m = _manifest()
    # 192x144 photo_noise: PNG ~60-80 KB, JPEG ~25 KB — both land in small.
    m.entries[0].output_formats = ["png", "jpeg"]
    outcome = build(m, tmp_path)
    assert outcome.ok, outcome.bucket_violations

    cases = load_cases(m, tmp_path, fmt_filter={"png"})
    assert all(c.fmt == "png" for c in cases)


def test_load_cases_bucket_filter(tmp_path: Path):
    m = _manifest()
    m.entries[0].bucket = Bucket.MEDIUM
    m.entries[0].width = 768
    m.entries[0].height = 576
    build(m, tmp_path)

    cases = load_cases(m, tmp_path, bucket_filter="small")
    assert all(c.bucket == "small" for c in cases)
    assert {c.name for c in cases} == {"photo_b"}


def test_load_cases_tag_filter(tmp_path: Path):
    m = _manifest()
    build(m, tmp_path)
    cases = load_cases(m, tmp_path, tag_filter="noise")
    assert {c.name for c in cases} == {"photo_b"}


def test_load_cases_preset_filter(tmp_path: Path):
    m = _manifest()
    build(m, tmp_path)
    cases = load_cases(m, tmp_path, preset_filter={"high"})
    assert all(c.preset == "high" for c in cases)
    assert len(cases) == 2


def test_load_cases_rejects_unknown_preset(tmp_path: Path):
    m = _manifest()
    build(m, tmp_path)
    with pytest.raises(ValueError, match="unknown presets"):
        load_cases(m, tmp_path, preset_filter={"hyper"})


def test_load_cases_raises_when_corpus_file_missing(tmp_path: Path):
    """If the corpus hasn't been built yet, fail fast with an actionable
    message — silent skip would produce empty benchmarks."""
    m = _manifest()
    with pytest.raises(CorpusFileMissing, match="Run `python -m bench.corpus build"):
        load_cases(m, tmp_path)


def test_case_load_returns_input_bytes(tmp_path: Path):
    m = _manifest()
    build(m, tmp_path)
    cases = load_cases(m, tmp_path, preset_filter={"high"})
    raw = cases[0].load()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(raw) == cases[0].input_size

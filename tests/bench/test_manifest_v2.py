"""Tests for ManifestEntry.source (SourceSpec) and v1/v2 schema compatibility."""

from __future__ import annotations

import json

import pytest

from bench.corpus.manifest import (
    Bucket,
    Manifest,
    ManifestEntry,
    ManifestSchemaError,
    SourceSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_entry(**overrides) -> ManifestEntry:
    base = dict(
        name="synth",
        bucket=Bucket.SMALL,
        content_kind="photo_gradient",
        seed=1,
        width=64,
        height=64,
        output_formats=["png"],
    )
    base.update(overrides)
    return ManifestEntry(**base)


def _fetched_entry(**overrides) -> ManifestEntry:
    base = dict(
        name="fetched",
        bucket=Bucket.MEDIUM,
        content_kind="fetched_photo",
        seed=0,
        width=768,
        height=512,
        output_formats=["png", "jpeg"],
        tags=["photo", "real_world", "kodak"],
        source=SourceSpec(
            url="https://example.com/img.png",
            sha256="a" * 64,
            license="Public domain",
            attribution="Test Corp",
            notes="test image",
        ),
    )
    base.update(overrides)
    return ManifestEntry(**base)


# ---------------------------------------------------------------------------
# SourceSpec tests
# ---------------------------------------------------------------------------


def test_source_spec_round_trips_with_notes() -> None:
    spec = SourceSpec(
        url="https://example.com/img.png",
        sha256="b" * 64,
        license="MIT",
        attribution="Alice",
        notes="some notes",
    )
    raw = spec.to_json()
    restored = SourceSpec.from_json(raw)
    assert restored.url == spec.url
    assert restored.sha256 == spec.sha256
    assert restored.license == spec.license
    assert restored.attribution == spec.attribution
    assert restored.notes == spec.notes


def test_source_spec_to_json_omits_empty_notes() -> None:
    """SourceSpec with no notes must not write the 'notes' key to JSON."""
    spec = SourceSpec(
        url="https://example.com/img.png",
        sha256="c" * 64,
        license="CC-BY",
        attribution="Bob",
    )
    raw = spec.to_json()
    assert "notes" not in raw


def test_source_spec_to_json_includes_notes_when_set() -> None:
    spec = SourceSpec(
        url="https://example.com/img.png",
        sha256="d" * 64,
        license="CC-BY",
        attribution="Bob",
        notes="something",
    )
    raw = spec.to_json()
    assert raw["notes"] == "something"


# ---------------------------------------------------------------------------
# v1 compatibility
# ---------------------------------------------------------------------------


def test_v1_manifest_loads_under_v2_schema() -> None:
    """A manifest_version=1 document (no source fields) must load successfully.

    All entries must have source=None after loading.
    """
    raw = {
        "manifest_version": 1,
        "manifest_name": "core",
        "library_versions": {"Pillow": "10.4.0"},
        "entries": [
            {
                "name": "photo_small",
                "bucket": "small",
                "content_kind": "photo_noise",
                "seed": 1,
                "width": 64,
                "height": 64,
                "output_formats": ["png"],
            },
            {
                "name": "graphic_small",
                "bucket": "small",
                "content_kind": "graphic_palette",
                "seed": 2,
                "width": 64,
                "height": 64,
                "output_formats": ["png"],
                "tags": ["graphic"],
            },
        ],
    }
    manifest = Manifest.from_json(raw)
    assert manifest.name == "core"
    assert len(manifest.entries) == 2
    for entry in manifest.entries:
        assert entry.source is None, f"{entry.name} unexpectedly has source set"


# ---------------------------------------------------------------------------
# v2 round-trip
# ---------------------------------------------------------------------------


def test_v2_manifest_with_mixed_entries_roundtrips() -> None:
    """A v2 manifest mixing synthesized and fetched entries must round-trip
    through to_json() → from_json() preserving all fields exactly."""
    m = Manifest(
        name="mixed",
        library_versions={"Pillow": "10.4.0"},
        entries=[
            _synth_entry(name="s1"),
            _fetched_entry(name="f1"),
        ],
    )

    raw = json.loads(json.dumps(m.to_json()))
    restored = Manifest.from_json(raw)

    assert restored.name == "mixed"
    assert len(restored.entries) == 2

    s = restored.entries[0]
    assert s.name == "s1"
    assert s.source is None
    assert s.content_kind == "photo_gradient"

    f = restored.entries[1]
    assert f.name == "f1"
    assert f.source is not None
    assert f.source.url == "https://example.com/img.png"
    assert f.source.sha256 == "a" * 64
    assert f.source.license == "Public domain"
    assert f.source.attribution == "Test Corp"
    assert f.source.notes == "test image"
    assert f.content_kind == "fetched_photo"
    assert "photo" in f.tags


def test_fetched_entry_to_json_includes_source_field() -> None:
    entry = _fetched_entry()
    raw = entry.to_json()
    assert "source" in raw
    assert raw["source"]["url"] == "https://example.com/img.png"
    assert raw["source"]["sha256"] == "a" * 64


def test_synth_entry_to_json_excludes_source_field() -> None:
    entry = _synth_entry()
    raw = entry.to_json()
    assert "source" not in raw


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_manifest_version_99_raises() -> None:
    """An unrecognised manifest_version must raise ManifestSchemaError."""
    with pytest.raises(ManifestSchemaError) as exc_info:
        Manifest.from_json(
            {
                "manifest_version": 99,
                "manifest_name": "x",
                "entries": [],
            }
        )
    assert "99" in str(exc_info.value)
    assert "expected 2" in str(exc_info.value)


def test_manifest_version_2_accepted() -> None:
    """manifest_version=2 is the primary version and must load without error."""
    m = Manifest.from_json(
        {
            "manifest_version": 2,
            "manifest_name": "v2",
            "library_versions": {},
            "entries": [_synth_entry().to_json()],
        }
    )
    assert m.manifest_version == 2
    assert len(m.entries) == 1

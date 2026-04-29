"""Manifest schema + pixel hash + verifier tests."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from bench.corpus.manifest import (
    MANIFEST_VERSION,
    Bucket,
    Manifest,
    ManifestEntry,
    ManifestSchemaError,
    bucket_for_size,
    normalized_mode,
    pixel_sha256,
    verify,
)


def _entry(**overrides) -> ManifestEntry:
    base = dict(
        name="t",
        bucket=Bucket.SMALL,
        content_kind="photo_gradient",
        seed=1,
        width=64,
        height=64,
        output_formats=["png"],
    )
    base.update(overrides)
    return ManifestEntry(**base)


def test_bucket_ranges_cover_size_axis():
    assert bucket_for_size(0) is Bucket.TINY
    assert bucket_for_size(10 * 1024 - 1) is Bucket.TINY
    assert bucket_for_size(10 * 1024) is Bucket.SMALL
    assert bucket_for_size(100 * 1024 - 1) is Bucket.SMALL
    assert bucket_for_size(100 * 1024) is Bucket.MEDIUM
    assert bucket_for_size(1024 * 1024) is Bucket.LARGE
    assert bucket_for_size(5 * 1024 * 1024) is Bucket.XLARGE
    assert bucket_for_size(50 * 1024 * 1024) is Bucket.XLARGE


def test_pixel_sha256_is_deterministic_for_same_pixels():
    a = Image.new("RGB", (32, 32), (10, 20, 30))
    b = Image.new("RGB", (32, 32), (10, 20, 30))
    assert pixel_sha256(a) == pixel_sha256(b)


def test_pixel_sha256_changes_when_pixels_change():
    a = Image.new("RGB", (32, 32), (10, 20, 30))
    b = Image.new("RGB", (32, 32), (10, 20, 31))
    assert pixel_sha256(a) != pixel_sha256(b)


def test_pixel_sha256_changes_when_size_changes():
    a = Image.new("RGB", (32, 32), (10, 20, 30))
    b = Image.new("RGB", (32, 33), (10, 20, 30))
    assert pixel_sha256(a) != pixel_sha256(b)


def test_pixel_sha256_changes_with_alpha_channel():
    rgb = Image.new("RGB", (16, 16), (255, 0, 0))
    rgba = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    assert pixel_sha256(rgb) != pixel_sha256(rgba)


def test_normalized_mode_expands_palette():
    palette = Image.new("P", (8, 8))
    assert normalized_mode(palette) in {"RGB", "RGBA"}


def test_normalized_mode_promotes_bilevel_to_grayscale():
    bilevel = Image.new("1", (8, 8), 1)
    assert normalized_mode(bilevel) == "L"


def test_manifest_round_trip_preserves_entries():
    m = Manifest(
        name="core",
        library_versions={"Pillow": "10.4.0"},
        entries=[
            _entry(name="a", expected_pixel_sha256="abc"),
            _entry(name="b", tags=["pathological"], output_formats=["png", "webp"]),
        ],
    )
    raw = json.loads(json.dumps(m.to_json()))
    restored = Manifest.from_json(raw)

    assert restored.name == "core"
    assert restored.library_versions == {"Pillow": "10.4.0"}
    assert [e.name for e in restored.entries] == ["a", "b"]
    assert restored.entries[1].tags == ["pathological"]


def test_manifest_load_save_round_trip(tmp_path):
    m = Manifest(name="core", library_versions={}, entries=[_entry(name="x")])
    path = tmp_path / "core.json"
    m.save(path)
    loaded = Manifest.load(path)
    assert [e.name for e in loaded.entries] == ["x"]


def test_manifest_rejects_wrong_schema_version():
    with pytest.raises(ManifestSchemaError):
        Manifest.from_json({"manifest_version": 99, "manifest_name": "x", "entries": []})


def test_manifest_rejects_missing_required_fields():
    with pytest.raises(ManifestSchemaError):
        Manifest.from_json(
            {
                "manifest_version": MANIFEST_VERSION,
                "manifest_name": "x",
                "entries": [{"name": "incomplete"}],
            }
        )


def test_manifest_filter_by_bucket_and_tag():
    m = Manifest(
        name="x",
        library_versions={},
        entries=[
            _entry(name="a", bucket=Bucket.SMALL, tags=["photo"]),
            _entry(name="b", bucket=Bucket.LARGE, tags=["photo"]),
            _entry(name="c", bucket=Bucket.SMALL, tags=["graphic"]),
        ],
    )
    assert [e.name for e in m.filter(bucket="small")] == ["a", "c"]
    assert [e.name for e in m.filter(tag="photo")] == ["a", "b"]
    assert [e.name for e in m.filter(bucket="small", tag="photo")] == ["a"]


def test_atomic_write_does_not_corrupt_on_failure(tmp_path, monkeypatch):
    """If json.dump raises mid-write, the original file must still be intact."""

    path = tmp_path / "m.json"
    Manifest(name="orig", library_versions={}, entries=[]).save(path)

    original_text = path.read_text()

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("bench.corpus.manifest.json.dump", boom)
    with pytest.raises(RuntimeError):
        Manifest(name="new", library_versions={}, entries=[]).save(path)

    assert path.read_text() == original_text


def _stub_synth_red(entry: ManifestEntry) -> Image.Image:
    return Image.new("RGB", (entry.width, entry.height), (255, 0, 0))


def _stub_synth_blue(entry: ManifestEntry) -> Image.Image:
    return Image.new("RGB", (entry.width, entry.height), (0, 0, 255))


def test_verify_passes_when_pixel_hash_matches():
    entry = _entry(name="red", width=8, height=8)
    entry.expected_pixel_sha256 = pixel_sha256(_stub_synth_red(entry))
    m = Manifest(name="x", library_versions={}, entries=[entry])

    result = verify(m, _stub_synth_red)
    assert result.ok
    assert result.exit_code == 0


def test_verify_reports_mismatch_when_synth_drifts():
    entry = _entry(name="red", width=8, height=8)
    entry.expected_pixel_sha256 = pixel_sha256(_stub_synth_red(entry))
    m = Manifest(name="x", library_versions={}, entries=[entry])

    result = verify(m, _stub_synth_blue)
    assert not result.ok
    assert result.exit_code == 1
    assert "red" in result.mismatches[0]


def test_verify_reports_missing_hash_separately_from_mismatch():
    entry = _entry(name="red", width=8, height=8)
    m = Manifest(name="x", library_versions={}, entries=[entry])

    result = verify(m, _stub_synth_red)
    assert not result.ok
    assert result.exit_code == 2
    assert "red" in result.missing[0]


def test_verify_reports_synth_failure_as_schema_error():
    entry = _entry(name="boom", width=8, height=8)
    entry.expected_pixel_sha256 = "x" * 64
    m = Manifest(name="x", library_versions={}, entries=[entry])

    def explode(_e):
        raise ValueError("unknown content_kind")

    result = verify(m, explode)
    assert result.exit_code == 3
    assert "boom" in result.schema_errors[0]

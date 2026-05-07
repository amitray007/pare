"""End-to-end tests for the corpus builder + CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bench.corpus.builder import BuildOutcome, build, file_path, reseal_manifest
from bench.corpus.cli import main as cli_main
from bench.corpus.manifest import Bucket, Manifest, ManifestEntry, pixel_sha256
from bench.corpus.synthesis import synthesize


def _tiny_manifest() -> Manifest:
    """Two-entry manifest sized to land predictably in the small bucket.

    Uses incompressible noise content so the PNG output size stays
    inside `small` regardless of encoder version. Each entry declares a
    single output format to keep the build matrix predictable.
    """
    return Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="photo_small",
                bucket=Bucket.SMALL,
                content_kind="photo_noise",
                seed=1,
                width=192,
                height=144,
                output_formats=["png"],
                tags=["photo"],
            ),
            ManifestEntry(
                name="anim_small",
                bucket=Bucket.SMALL,
                content_kind="animated_translation",
                seed=2,
                width=96,
                height=72,
                output_formats=["apng"],
                tags=["animated"],
            ),
        ],
    )


def test_file_path_layout(tmp_path: Path):
    entry = ManifestEntry(
        name="x",
        bucket=Bucket.MEDIUM,
        content_kind="photo_perlin",
        seed=0,
        width=128,
        height=128,
        output_formats=["png"],
    )
    p = file_path(tmp_path, entry, "png")
    assert p == tmp_path / "medium" / "png" / "x.png"


def test_build_writes_files_and_skips_existing(tmp_path: Path):
    m = _tiny_manifest()
    outcome = build(m, tmp_path)
    assert outcome.ok, outcome.bucket_violations
    assert all(p.exists() for p in outcome.written)
    assert len(outcome.written) == 2

    # Second build skips everything
    outcome2 = build(m, tmp_path)
    assert outcome2.written == []
    assert len(outcome2.skipped) == 2


def test_build_force_rewrites_files(tmp_path: Path):
    m = _tiny_manifest()
    build(m, tmp_path)
    outcome = build(m, tmp_path, force=True)
    assert outcome.skipped == []
    assert len(outcome.written) == 2


def test_build_format_filter(tmp_path: Path):
    m = _tiny_manifest()
    outcome = build(m, tmp_path, formats_filter={"png"})
    assert all(p.suffix == ".png" for p in outcome.written)


def test_build_bucket_filter(tmp_path: Path):
    m = Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="a",
                bucket=Bucket.SMALL,
                content_kind="photo_perlin",
                seed=1,
                width=128,
                height=96,
                output_formats=["png"],
            ),
            ManifestEntry(
                name="b",
                bucket=Bucket.MEDIUM,
                content_kind="photo_perlin",
                seed=2,
                width=512,
                height=384,
                output_formats=["png"],
            ),
        ],
    )
    outcome = build(m, tmp_path, bucket_filter="small")
    assert len(outcome.written) == 1
    assert "/small/" in str(outcome.written[0])


def test_build_reports_bucket_violations(tmp_path: Path):
    """An entry declaring bucket=tiny while synthesizing at 1024×768 must
    fail validation, not silently produce a misclassified file."""
    m = Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="oversized",
                bucket=Bucket.TINY,
                content_kind="photo_perlin",
                seed=1,
                width=1024,
                height=768,
                output_formats=["png"],
            ),
        ],
    )
    outcome = build(m, tmp_path)
    assert not outcome.ok
    assert any("bucket violation" in v for v in outcome.bucket_violations)


def test_build_skips_unsupported_formats_with_message(tmp_path: Path, monkeypatch):
    """If an encoder is unavailable, the build should skip that format
    and record it in `format_skipped`, not abort."""
    m = _tiny_manifest()

    from bench.corpus import builder as builder_mod

    # Pretend only WEBP is encodable: PNG and APNG entries should be skipped
    monkeypatch.setattr(builder_mod, "supported_formats", lambda: {"webp"})
    outcome = build(m, tmp_path)
    assert outcome.written == []
    assert len(outcome.format_skipped) == 2
    assert all("encoder unavailable" in s for s in outcome.format_skipped)


def test_reseal_manifest_populates_pixel_hashes():
    m = _tiny_manifest()
    sealed = reseal_manifest(m)
    for entry, sealed_entry in zip(m.entries, sealed.entries):
        assert sealed_entry.expected_pixel_sha256 is not None
        # Sealing actually computes the right hash
        expected = pixel_sha256(synthesize(entry))
        assert sealed_entry.expected_pixel_sha256 == expected


def test_reseal_does_not_mutate_input_manifest():
    m = _tiny_manifest()
    reseal_manifest(m)
    for entry in m.entries:
        assert entry.expected_pixel_sha256 is None


def test_cli_list_runs_against_core_manifest(capsys):
    rc = cli_main(["list", "--manifest", "core"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "manifest=core" in out
    assert "supported_formats=" in out


def test_cli_verify_returns_2_on_unsealed_manifest(tmp_path: Path, capsys):
    """Newly-authored manifest has no pixel hashes — exit 2."""
    m = _tiny_manifest()
    path = tmp_path / "t.json"
    m.save(path)
    rc = cli_main(["verify", "--manifest", str(path)])
    assert rc == 2


def test_cli_seal_then_verify_returns_0(tmp_path: Path, capsys):
    m = _tiny_manifest()
    path = tmp_path / "t.json"
    m.save(path)

    rc = cli_main(["build", "--manifest", str(path), "--out", str(tmp_path / "corpus"), "--seal"])
    assert rc == 0

    rc = cli_main(["verify", "--manifest", str(path)])
    assert rc == 0


def test_cli_verify_returns_1_on_seed_drift(tmp_path: Path):
    """Tampering with the manifest seed after sealing causes pixel hashes
    to drift; verify must surface this."""
    m = _tiny_manifest()
    path = tmp_path / "t.json"
    m.save(path)

    cli_main(["build", "--manifest", str(path), "--out", str(tmp_path / "corpus"), "--seal"])

    # Mutate seed without resealing
    raw = json.loads(path.read_text())
    raw["entries"][0]["seed"] = 999
    path.write_text(json.dumps(raw))

    rc = cli_main(["verify", "--manifest", str(path)])
    assert rc == 1


def test_cli_unknown_manifest_exits_with_clear_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["verify", "--manifest", "does-not-exist"])
    assert "manifest not found" in str(exc.value)


def test_outcome_ok_property():
    outcome = BuildOutcome()
    assert outcome.ok
    outcome.bucket_violations.append("x")
    assert not outcome.ok


def test_logging_does_not_fail_at_info_level(tmp_path: Path):
    """Smoke: --verbose flag through main()."""
    logging.getLogger("bench.corpus").handlers = []  # reset between tests
    rc = cli_main(["--verbose", "list", "--manifest", "core"])
    assert rc == 0

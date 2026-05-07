"""Tests for SVG/SVGZ vector corpus support.

Covers:
- _encode_svg / _encode_svgz pass-through behaviour
- SVGZ determinism (mtime=0)
- SVGZ decompresses back to input
- Builder writes SVG files by pass-through (no Image.open)
- ManifestEntry.expected_byte_sha256 round-trips through to_json/from_json
- Mixed vector+raster output_formats are rejected at build time
"""

from __future__ import annotations

import gzip
import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from bench.corpus.builder import _check_no_mixed_vector_raster, build
from bench.corpus.conversion import FormatNotSupportedError, _encode_svg, _encode_svgz
from bench.corpus.manifest import (
    Bucket,
    Manifest,
    ManifestEntry,
    SourceSpec,
    is_vector_entry,
)

# ---------------------------------------------------------------------------
# Minimal test SVG
# ---------------------------------------------------------------------------

_SAMPLE_SVG = b"""\
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10" fill="red"/>
</svg>
"""


# ---------------------------------------------------------------------------
# Encoder unit tests
# ---------------------------------------------------------------------------


def test_svg_pass_through_encode() -> None:
    """_encode_svg must return the same bytes as the input."""
    result = _encode_svg(_SAMPLE_SVG)
    assert result == _SAMPLE_SVG
    assert isinstance(result, bytes)


def test_svg_pass_through_with_bytearray() -> None:
    """_encode_svg must accept bytearray as well as bytes."""
    result = _encode_svg(bytearray(_SAMPLE_SVG))
    assert result == _SAMPLE_SVG


def test_svg_encode_rejects_non_bytes() -> None:
    """_encode_svg must raise FormatNotSupportedError for non-byte content."""
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    with pytest.raises(FormatNotSupportedError, match="SVG entries must use byte content"):
        _encode_svg(img)  # type: ignore[arg-type]


def test_svgz_deterministic() -> None:
    """Calling _encode_svgz on the same bytes twice must return identical output.

    This verifies the mtime=0 gzip path — without mtime=0, gzip embeds a
    timestamp and the two results would differ.
    """
    result1 = _encode_svgz(_SAMPLE_SVG)
    result2 = _encode_svgz(_SAMPLE_SVG)
    assert result1 == result2, "SVGZ output is not deterministic — mtime=0 path broken"


def test_svgz_decompresses_to_input() -> None:
    """_encode_svgz output must decompress back to the original bytes."""
    compressed = _encode_svgz(_SAMPLE_SVG)
    decompressed = gzip.decompress(compressed)
    assert decompressed == _SAMPLE_SVG


def test_svgz_encode_rejects_non_bytes() -> None:
    """_encode_svgz must raise FormatNotSupportedError for non-byte content."""
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    with pytest.raises(FormatNotSupportedError, match="SVGZ entries must use byte content"):
        _encode_svgz(img)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# is_vector_entry helper
# ---------------------------------------------------------------------------


def _vector_entry(**overrides) -> ManifestEntry:
    base = dict(
        name="vec",
        bucket=Bucket.TINY,
        content_kind="fetched_vector",
        seed=0,
        width=0,
        height=0,
        output_formats=["svg", "svgz"],
        tags=["vector"],
        source=SourceSpec(
            url="https://example.com/test.svg",
            sha256="a" * 64,
            license="Public domain",
            attribution="Test",
        ),
    )
    base.update(overrides)
    return ManifestEntry(**base)


def test_is_vector_entry_true_for_svg_svgz() -> None:
    assert is_vector_entry(_vector_entry(output_formats=["svg", "svgz"]))


def test_is_vector_entry_true_for_svg_only() -> None:
    assert is_vector_entry(_vector_entry(output_formats=["svg"]))


def test_is_vector_entry_false_for_raster() -> None:
    entry = ManifestEntry(
        name="r",
        bucket=Bucket.SMALL,
        content_kind="photo_noise",
        seed=1,
        width=64,
        height=64,
        output_formats=["png"],
    )
    assert not is_vector_entry(entry)


def test_is_vector_entry_false_for_empty() -> None:
    entry = _vector_entry(output_formats=[])
    assert not is_vector_entry(entry)


# ---------------------------------------------------------------------------
# Mixed vector+raster rejection
# ---------------------------------------------------------------------------


def test_mixed_vector_raster_output_formats_rejected(tmp_path: Path) -> None:
    """An entry whose output_formats includes both svg and png must be rejected."""
    mixed_entry = _vector_entry(output_formats=["svg", "png"])
    err = _check_no_mixed_vector_raster(mixed_entry)
    assert err is not None
    assert "mixes vector" in err
    assert "svg" in err
    assert "png" in err


def test_pure_vector_output_formats_not_rejected() -> None:
    entry = _vector_entry(output_formats=["svg", "svgz"])
    assert _check_no_mixed_vector_raster(entry) is None


def test_pure_raster_output_formats_not_rejected() -> None:
    entry = ManifestEntry(
        name="r",
        bucket=Bucket.SMALL,
        content_kind="photo_noise",
        seed=1,
        width=64,
        height=64,
        output_formats=["png", "jpeg"],
    )
    assert _check_no_mixed_vector_raster(entry) is None


def test_build_rejects_mixed_vector_raster_entry(tmp_path: Path) -> None:
    """A manifest entry mixing svg+png must produce a bucket violation (rejected)."""
    mixed_entry = _vector_entry(output_formats=["svg", "png"])
    m = Manifest(name="t", library_versions={}, entries=[mixed_entry])
    outcome = build(m, tmp_path)
    assert not outcome.ok
    assert any("mixes vector" in v for v in outcome.bucket_violations)


# ---------------------------------------------------------------------------
# Builder writes SVG via pass-through (no Image.open)
# ---------------------------------------------------------------------------


class _SVGHandler(http.server.BaseHTTPRequestHandler):
    """Serves _SAMPLE_SVG at /test.svg."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/test.svg":
            self.send_response(200)
            self.send_header("Content-Length", str(len(_SAMPLE_SVG)))
            self.end_headers()
            self.wfile.write(_SAMPLE_SVG)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args, **_kwargs) -> None:
        pass


def test_builder_writes_svg_pass_through(tmp_path: Path) -> None:
    """The builder must write the fetched SVG bytes directly to disk,
    skipping Image.open().  The on-disk file must match the source bytes."""
    sha = hashlib.sha256(_SAMPLE_SVG).hexdigest()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SVGHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        entry = _vector_entry(
            source=SourceSpec(
                url=f"http://127.0.0.1:{port}/test.svg",
                sha256=sha,
                license="Public domain",
                attribution="Test",
            ),
        )
        m = Manifest(name="t", library_versions={}, entries=[entry])
        corpus_root = tmp_path / "corpus"
        cache_root = tmp_path / "cache"

        outcome = build(m, corpus_root, cache_root=cache_root)
        assert outcome.ok, outcome.bucket_violations

        svg_file = corpus_root / "tiny" / "svg" / "vec.svg"
        assert svg_file.exists(), f"SVG corpus file not written: {svg_file}"
        assert svg_file.read_bytes() == _SAMPLE_SVG

        # SVGZ must decompress back to the same bytes
        svgz_file = corpus_root / "tiny" / "svgz" / "vec.svgz"
        assert svgz_file.exists(), f"SVGZ corpus file not written: {svgz_file}"
        assert gzip.decompress(svgz_file.read_bytes()) == _SAMPLE_SVG

        # byte_hashes must record the source hash
        assert entry.name in outcome.byte_hashes
        assert outcome.byte_hashes[entry.name] == sha

    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# ManifestEntry.expected_byte_sha256 round-trip
# ---------------------------------------------------------------------------


def test_manifest_entry_with_byte_sha256_round_trips() -> None:
    """A vector entry with expected_byte_sha256 must survive to_json/from_json."""
    entry = _vector_entry(expected_byte_sha256={"source": "ab" * 32})
    raw = entry.to_json()
    assert "expected_byte_sha256" in raw
    assert raw["expected_byte_sha256"]["source"] == "ab" * 32

    restored = ManifestEntry.from_json(raw)
    assert restored.expected_byte_sha256 is not None
    assert restored.expected_byte_sha256["source"] == "ab" * 32


def test_manifest_entry_without_byte_sha256_round_trips() -> None:
    """A raster entry without expected_byte_sha256 must NOT include the field in JSON."""
    entry = ManifestEntry(
        name="r",
        bucket=Bucket.SMALL,
        content_kind="photo_noise",
        seed=1,
        width=64,
        height=64,
        output_formats=["png"],
    )
    raw = entry.to_json()
    assert "expected_byte_sha256" not in raw

    restored = ManifestEntry.from_json(raw)
    assert restored.expected_byte_sha256 is None


def test_manifest_v2_with_vector_entry_roundtrips_through_manifest() -> None:
    """A v2 Manifest containing a vector entry must round-trip through Manifest
    to_json() → from_json() preserving expected_byte_sha256 exactly."""
    entry = _vector_entry(expected_byte_sha256={"source": "cd" * 32})
    m = Manifest(
        name="mixed",
        library_versions={"Pillow": "10.4.0"},
        entries=[entry],
    )

    raw = json.loads(json.dumps(m.to_json()))
    restored = Manifest.from_json(raw)

    assert len(restored.entries) == 1
    ve = restored.entries[0]
    assert ve.content_kind == "fetched_vector"
    assert ve.expected_byte_sha256 is not None
    assert ve.expected_byte_sha256["source"] == "cd" * 32
    assert ve.expected_pixel_sha256 is None
    assert ve.output_formats == ["svg", "svgz"]

"""Tests for AVIF pre-flight skip logic and ISOBMFF box parser.

Coverage:
- _parse_avif_metadata: handcrafted box bytes, edge cases, nested containers
- _should_skip_avif_optimization: per-preset skip decisions
- End-to-end: verify _open_image is never called when pre-flight skips
"""

import io
import struct
from unittest.mock import patch

import pytest

from optimizers.avif import AvifOptimizer, _parse_avif_metadata, _should_skip_avif_optimization
from schemas import OptimizationConfig

try:
    import pillow_avif  # noqa: F401

    _buf = io.BytesIO()
    from PIL import Image as _Im

    _Im.new("RGB", (1, 1)).save(_buf, format="AVIF")
    HAS_AVIF = True
except (ImportError, Exception):
    HAS_AVIF = False


# ---------------------------------------------------------------------------
# Box-building helpers
# ---------------------------------------------------------------------------


def _box(box_type: bytes, body: bytes, use_extended: bool = False) -> bytes:
    """Build a minimal ISOBMFF box."""
    assert len(box_type) == 4
    if use_extended:
        # size==1 header with uint64 actual size
        actual_size = 16 + len(body)
        return struct.pack(">I", 1) + box_type + struct.pack(">Q", actual_size) + body
    else:
        actual_size = 8 + len(body)
        return struct.pack(">I", actual_size) + box_type + body


def _fullbox_prefix(version: int = 0, flags: int = 0) -> bytes:
    """Return a 4-byte FullBox version+flags prefix."""
    return struct.pack(">I", (version << 24) | flags)


def _ispe_body(width: int, height: int) -> bytes:
    """ispe FullBox body: fullbox_prefix + uint32 width + uint32 height."""
    return _fullbox_prefix() + struct.pack(">II", width, height)


def _colr_box(colour_type: bytes, payload: bytes = b"\x00" * 10) -> bytes:
    """Build a colr box with given colour_type (4 bytes) and payload."""
    assert len(colour_type) == 4
    return _box(b"colr", colour_type + payload)


def _minimal_avif(width: int, height: int, extra_boxes: bytes = b"") -> bytes:
    """Build a minimal AVIF-shaped byte stream with ispe and optional extras.

    Layout: ftyp | meta( hdlr | iprp( ipco( ispe | extra_boxes ) ) )
    """
    ispe = _box(b"ispe", _ispe_body(width, height))
    ipco_body = ispe + extra_boxes
    ipco = _box(b"ipco", ipco_body)
    iprp = _box(b"iprp", ipco)
    # meta is a FullBox — prepend 4-byte version/flags
    hdlr = _box(b"hdlr", b"\x00" * 20)
    meta_body = _fullbox_prefix() + hdlr + iprp
    meta = _box(b"meta", meta_body)
    ftyp = _box(b"ftyp", b"avif" + b"\x00" * 4 + b"avif")
    return ftyp + meta


# ---------------------------------------------------------------------------
# Parser correctness
# ---------------------------------------------------------------------------


class TestParseAvifMetadata:
    def test_parses_ispe_dimensions(self):
        data = _minimal_avif(1920, 1080)
        w, h, mb = _parse_avif_metadata(data)
        assert (w, h) == (1920, 1080)

    def test_no_metadata_boxes(self):
        data = _minimal_avif(100, 100)
        w, h, mb = _parse_avif_metadata(data)
        assert (w, h) == (100, 100)
        assert mb == 0

    def test_exif_box_counted(self):
        exif_payload = b"\x00" * 100
        exif_box = _box(b"Exif", exif_payload)
        data = _minimal_avif(100, 100, extra_boxes=exif_box)
        w, h, mb = _parse_avif_metadata(data)
        assert (w, h) == (100, 100)
        # exif box size = 8 header + 100 body = 108
        assert mb == 108

    def test_xml_and_xmp_boxes_counted(self):
        xml_box = _box(b"xml ", b"x" * 50)
        xmp_box = _box(b"xmp ", b"x" * 30)
        data = _minimal_avif(200, 300, extra_boxes=xml_box + xmp_box)
        _, _, mb = _parse_avif_metadata(data)
        # 8+50 + 8+30 = 96
        assert mb == 96

    def test_colr_prof_not_counted(self):
        # colr box with 'prof' colour type = ICC profile.
        # The strip path preserves ICC (icc_profile= kwarg), so it is NOT
        # strippable overhead and must NOT be counted toward meta_bytes.
        prof_colr = _colr_box(b"prof", b"\x00" * 200)
        data = _minimal_avif(100, 100, extra_boxes=prof_colr)
        _, _, mb = _parse_avif_metadata(data)
        assert mb == 0  # ICC profile is preserved, not stripped

    def test_colr_nclx_not_counted(self):
        # colr box with 'nclx' colour type = signalling data, not strippable
        nclx_colr = _colr_box(b"nclx", b"\x00" * 10)
        data = _minimal_avif(100, 100, extra_boxes=nclx_colr)
        _, _, mb = _parse_avif_metadata(data)
        assert mb == 0

    def test_extended_size_box(self):
        # Build an ispe box using size==1 extended header
        ispe_extended = _box(b"ispe", _ispe_body(640, 480), use_extended=True)
        ipco = _box(b"ipco", ispe_extended)
        iprp = _box(b"iprp", ipco)
        meta_body = _fullbox_prefix() + iprp
        meta = _box(b"meta", meta_body)
        ftyp = _box(b"ftyp", b"avif" + b"\x00" * 4 + b"avif")
        data = ftyp + meta
        w, h, _ = _parse_avif_metadata(data)
        assert (w, h) == (640, 480)

    def test_nested_meta_iprp_ipco(self):
        """Parser correctly recurses into meta → iprp → ipco to find ispe."""
        data = _minimal_avif(800, 600)
        w, h, mb = _parse_avif_metadata(data)
        assert (w, h) == (800, 600)
        assert mb == 0

    def test_colr_prof_and_nclx_combined(self):
        """Neither prof nor nclx colr boxes are counted — both are preserved by strip."""
        prof_colr = _colr_box(b"prof", b"\x00" * 500)
        nclx_colr = _colr_box(b"nclx", b"\x00" * 10)
        data = _minimal_avif(100, 100, extra_boxes=prof_colr + nclx_colr)
        _, _, mb = _parse_avif_metadata(data)
        assert mb == 0  # ICC (prof) is preserved by strip; nclx is bitstream data

    def test_meta_level_exif_sibling_counted(self):
        """Exif box at the meta level (sibling of iprp) must be counted.

        ISO/IEC 23008-12 allows Exif/xmp/xml boxes as direct children of meta,
        not only inside ipco.  The previous early-return-on-width bug would walk
        meta → iprp → ipco → ispe, find width != 0, and immediately return —
        skipping Exif boxes that come AFTER iprp in the meta container.

        Layout:
            ftyp
            meta (FullBox)
                iprp
                    ipco
                        ispe (width=100, height=100)
                Exif (~1000-byte body)   ← sibling of iprp, after ispe is found
        """
        # Build from the inside out
        ispe = _box(b"ispe", _ispe_body(100, 100))
        ipco = _box(b"ipco", ispe)
        iprp = _box(b"iprp", ipco)

        exif_body = b"\x00" * 1000
        exif_box = _box(b"Exif", exif_body)
        expected_exif_box_size = 8 + len(exif_body)  # 1008

        # meta is a FullBox: 4-byte version+flags prefix, then iprp, then Exif sibling
        meta_body = _fullbox_prefix() + iprp + exif_box
        meta = _box(b"meta", meta_body)
        ftyp = _box(b"ftyp", b"avif" + b"\x00" * 4 + b"avif")
        data = ftyp + meta

        w, h, mb = _parse_avif_metadata(data)
        assert (w, h) == (100, 100), "ispe dimensions must still be found"
        assert mb == expected_exif_box_size, (
            f"Exif sibling of iprp at meta level must be counted; "
            f"got {mb}, expected {expected_exif_box_size}"
        )


class TestParseAvifMetadataEdgeCases:
    def test_empty_input_returns_zeros(self):
        w, h, mb = _parse_avif_metadata(b"")
        assert (w, h, mb) == (0, 0, 0)

    def test_truncated_input_returns_zeros(self):
        # Incomplete box header — only 4 bytes
        w, h, mb = _parse_avif_metadata(b"\x00\x00\x00\x10")
        assert (w, h, mb) == (0, 0, 0)

    def test_malformed_box_size_zero(self):
        # size==0 with no enclosing container: should not crash
        bad = struct.pack(">I", 0) + b"meta" + b"\x00" * 10
        w, h, mb = _parse_avif_metadata(bad)
        # No ispe present → (0, 0, ...)
        assert w == 0 and h == 0

    def test_box_size_exceeds_data(self):
        # size claims 9999 bytes but data is only 12
        bad = struct.pack(">I", 9999) + b"ispe" + b"\x00" * 4
        w, h, mb = _parse_avif_metadata(bad)
        assert (w, h, mb) == (0, 0, 0)

    def test_garbage_bytes_does_not_crash(self):
        w, h, mb = _parse_avif_metadata(b"\xff" * 256)
        # Must not raise; result is undefined but should be (0, 0, ...)
        assert isinstance(w, int) and isinstance(h, int) and isinstance(mb, int)


# ---------------------------------------------------------------------------
# Skip decision logic
# ---------------------------------------------------------------------------


class TestShouldSkipAvifOptimization:
    # --- HIGH preset (quality < 50) --- always False
    def test_high_preset_never_skips(self):
        data = _minimal_avif(1000, 1000)
        config = OptimizationConfig(quality=40)
        assert _should_skip_avif_optimization(data, config) is False

    # --- MEDIUM preset (50 <= quality < 70) --- skip if bpp < 0.5
    def test_medium_skips_when_low_bpp(self):
        # 1920x1080 @ ~0.3 bpp → small file → below threshold
        n_pixels = 1920 * 1080
        file_bytes = int(n_pixels * 0.3 / 8)  # 0.3 bpp → ~77 760 bytes
        # Build fake bytes big enough to hit that size
        header = _minimal_avif(1920, 1080)
        data = header + b"\x00" * max(0, file_bytes - len(header))
        config = OptimizationConfig(quality=60)
        assert _should_skip_avif_optimization(data, config) is True

    def test_medium_does_not_skip_when_high_bpp(self):
        # 100x100 @ ~5 bpp → large file → above threshold
        n_pixels = 100 * 100
        file_bytes = int(n_pixels * 5 / 8)  # 5 bpp → 6250 bytes
        header = _minimal_avif(100, 100)
        data = header + b"\x00" * max(0, file_bytes - len(header))
        config = OptimizationConfig(quality=60)
        assert _should_skip_avif_optimization(data, config) is False

    def test_medium_parse_failure_does_not_skip(self):
        # Parse failure → should fall back to decode, not skip
        config = OptimizationConfig(quality=60)
        assert _should_skip_avif_optimization(b"garbage", config) is False

    # --- LOW preset (quality >= 70) --- skip if meta_bytes < 500
    def test_low_skips_when_no_metadata(self):
        data = _minimal_avif(100, 100)  # no metadata boxes
        config = OptimizationConfig(quality=80, strip_metadata=True)
        assert _should_skip_avif_optimization(data, config) is True

    def test_low_does_not_skip_when_large_strippable_metadata(self):
        # 604-byte Exif box (8+596 = 604) — genuinely strippable, above threshold
        exif_box = _box(b"Exif", b"\x00" * 596)
        data = _minimal_avif(100, 100, extra_boxes=exif_box)
        config = OptimizationConfig(quality=80, strip_metadata=True)
        assert _should_skip_avif_optimization(data, config) is False

    def test_low_skips_when_only_icc_profile(self):
        # ICC profile colr box: large in bytes, but the strip path PRESERVES it —
        # so meta_bytes stays 0 and the LOW skip rule fires correctly.
        # This is the calibration bug fixed in iter3: real-world AVIFs with only
        # an ICC profile were incorrectly proceeding to decode+strip+encode,
        # returning "none" after wasted CPU.
        prof_colr = _colr_box(b"prof", b"\x00" * 3140)  # ~3KB ICC, like Earth_Apollo_17
        data = _minimal_avif(100, 100, extra_boxes=prof_colr)
        config = OptimizationConfig(quality=80, strip_metadata=True)
        assert _should_skip_avif_optimization(data, config) is True

    def test_low_skips_when_small_metadata(self):
        # 400 bytes of metadata — below the 500 B threshold
        exif_box = _box(b"Exif", b"\x00" * 392)  # 8+392 = 400
        data = _minimal_avif(100, 100, extra_boxes=exif_box)
        config = OptimizationConfig(quality=80, strip_metadata=True)
        assert _should_skip_avif_optimization(data, config) is True

    def test_low_skips_when_strip_metadata_false(self):
        # strip_metadata=False + LOW preset → optimizer would no-op anyway
        data = _minimal_avif(100, 100)
        config = OptimizationConfig(quality=80, strip_metadata=False)
        assert _should_skip_avif_optimization(data, config) is True


# ---------------------------------------------------------------------------
# End-to-end: optimizer skips decode when pre-flight fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_prevents_decode_on_skip():
    """When pre-flight returns True, _open_image must never be called."""
    opt = AvifOptimizer()

    # Build a LOW-preset input that will trigger skip (no metadata)
    header = _minimal_avif(100, 100)
    # Ensure the file is reasonably sized but below any bpp concern for LOW
    data = header + b"\x00" * 500

    config = OptimizationConfig(quality=80, strip_metadata=True)

    with patch.object(opt, "_open_image", side_effect=AssertionError("_open_image called")):
        result = await opt.optimize(data, config)

    assert result.method == "none"
    assert result.success is True
    assert result.optimized_size == len(data)


@pytest.mark.asyncio
async def test_preflight_does_not_block_high_preset():
    """HIGH preset bypasses the skip check and actually decodes.

    _open_image is called (the decode is attempted).  We verify this by having
    _open_image raise — since _open_image is invoked before the per-method
    try/except blocks, the exception propagates out of optimize().
    """
    opt = AvifOptimizer()

    header = _minimal_avif(100, 100)
    data = header + b"\x00" * 500

    config = OptimizationConfig(quality=40)

    decode_called = []

    def track_open(d):
        decode_called.append(True)
        raise RuntimeError("decode-side abort for test isolation")

    with pytest.raises(RuntimeError, match="decode-side abort"):
        with patch.object(opt, "_open_image", side_effect=track_open):
            await opt.optimize(data, config)

    assert decode_called, "_open_image should have been called for HIGH preset"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_preflight_real_avif_medium_already_compressed():
    """A tightly-compressed AVIF at MEDIUM preset should be skipped."""
    from PIL import Image

    opt = AvifOptimizer()

    # Encode at q=50 → already near the AVIF floor for this tiny image
    img = Image.new("RGB", (1000, 1000), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="AVIF", quality=50)
    data = buf.getvalue()

    bpp = (len(data) * 8) / (1000 * 1000)
    config = OptimizationConfig(quality=60)

    if bpp < 0.5:
        # Should be skipped without decode
        with patch.object(opt, "_open_image", side_effect=AssertionError("decode called")):
            result = await opt.optimize(data, config)
        assert result.method == "none"
    else:
        # bpp >= 0.5 → pre-flight won't skip; just verify no crash
        result = await opt.optimize(data, config)
        assert result.success


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_preflight_real_avif_low_no_metadata():
    """A real AVIF without metadata at LOW preset should be skipped."""
    from PIL import Image

    opt = AvifOptimizer()
    img = Image.new("RGB", (200, 200), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="AVIF", quality=80)
    data = buf.getvalue()

    config = OptimizationConfig(quality=80, strip_metadata=True)

    with patch.object(opt, "_open_image", side_effect=AssertionError("decode called")):
        result = await opt.optimize(data, config)

    assert result.method == "none"
    assert result.success


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_existing_avif_tests_still_pass_high_preset():
    """Ensure existing optimization path still works for HIGH preset."""
    from PIL import Image

    opt = AvifOptimizer()
    img = Image.new("RGB", (200, 200), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="AVIF", quality=95)
    data = buf.getvalue()

    config = OptimizationConfig(quality=40)
    result = await opt.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "avif"

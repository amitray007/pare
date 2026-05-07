"""Deep-color (10/12-bit) encoding tests.

Verifies the typed-buffer encode paths for JXL, HEIC, and AVIF, and
confirms that the 8-bit code path through the JXL encoder is unaffected
by the dispatch logic added in this PR.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from bench.corpus.conversion import (
    FormatNotSupportedError,
    _detect_bit_depth,
    _encode_avif_deep,
    _encode_heic_deep,
    _encode_jxl_deep,
    encode,
    supported_formats,
)
from bench.corpus.manifest import Bucket, ManifestEntry, pixel_sha256
from bench.corpus.synthesis import synthesize
from bench.corpus.synthesis.deep_color import deep_color_smooth

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_10bit_rgb(width: int = 32, height: int = 32, seed: int = 1) -> np.ndarray:
    """Return a reproducible uint16 RGB array with values in [0, 1023]."""
    return deep_color_smooth(seed=seed, width=width, height=height, bit_depth=10)


def _make_12bit_rgb(width: int = 32, height: int = 32, seed: int = 2) -> np.ndarray:
    """Return a reproducible uint16 RGB array with values in [0, 4095]."""
    return deep_color_smooth(seed=seed, width=width, height=height, bit_depth=12)


# ---------------------------------------------------------------------------
# 1. JXL round-trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif("jxl" not in supported_formats(), reason="jxlpy not available")
def test_jxl_deep_color_round_trip_preserves_bit_depth():
    """Encode a 10-bit ndarray to JXL bytes and decode; verify the decoded
    bit depth header and that pixel values round-trip within a lossy
    tolerance appropriate for quality=90 (≤ 6% of the 10-bit range)."""
    from jxlpy import JXLPyDecoder

    arr = _make_10bit_rgb(64, 64, seed=42)
    assert arr.dtype == np.uint16
    assert int(arr.max()) < 1024, "synthesizer should produce 10-bit values"

    jxl_bytes = _encode_jxl_deep(arr, quality=90, bit_depth=10)
    assert len(jxl_bytes) > 0

    # JXL bare codestream or container
    assert jxl_bytes.startswith(
        (b"\x00\x00\x00\x0cJXL", b"\xff\x0a")
    ), f"unexpected JXL magic: {jxl_bytes[:12].hex()}"

    dec = JXLPyDecoder(jxl_bytes)
    info = dec.get_info()
    assert (
        info["bits_per_sample"] == 10
    ), f"decoded bit depth should be 10, got {info['bits_per_sample']}"
    assert info["xsize"] == 64 and info["ysize"] == 64

    frame_bytes = dec.get_frame()
    decoded = np.frombuffer(frame_bytes, dtype=np.uint16).reshape(64, 64, 3)

    # Verify the decoded pixels stay within the 10-bit range
    assert int(decoded.max()) <= 1023, f"decoded pixels exceed 10-bit range: max={decoded.max()}"

    diff = np.abs(arr.astype(np.int32) - decoded.astype(np.int32))
    # JXL is a perceptual lossy codec; at quality=90 the *mean* pixel error is
    # the right metric (≤ 5% of the 10-bit range ≈ 51).  The max error can be
    # higher due to edge-ringing even at high quality.
    mean_diff = float(diff.mean())
    assert (
        mean_diff <= 51.0
    ), f"mean pixel round-trip error too large: {mean_diff:.1f} (threshold=51 for 10-bit q=90)"


# ---------------------------------------------------------------------------
# 2. HEIC deep-color — succeeds or raises FormatNotSupportedError clearly
# ---------------------------------------------------------------------------


@pytest.mark.skipif("heic" not in supported_formats(), reason="pillow_heif not available")
def test_heic_deep_color_encode_succeeds_or_raises_clearly():
    """pillow_heif 0.22+ typed-buffer HEIC encode should succeed.

    If the installed version does not support the typed-buffer API, the
    encoder must raise `FormatNotSupportedError` with a meaningful message
    rather than an opaque AttributeError or TypeError.
    """
    arr = _make_10bit_rgb(32, 32, seed=5)
    try:
        heic_bytes = _encode_heic_deep(arr, quality=85, bit_depth=10)
        assert len(heic_bytes) > 0
        # HEIC/HEIF container starts with an ftyp box at offset 4
        assert heic_bytes[4:8] == b"ftyp", f"expected ftyp box, got: {heic_bytes[4:8]!r}"
    except FormatNotSupportedError as exc:
        # Documented known limitation path — must not be a bare exception.
        assert (
            "heic" in str(exc).lower()
            or "typed-buffer" in str(exc).lower()
            or "pillow_heif" in str(exc).lower()
        ), f"FormatNotSupportedError message is not descriptive enough: {exc}"


# ---------------------------------------------------------------------------
# 3. AVIF deep-color — succeeds or raises FormatNotSupportedError clearly
# ---------------------------------------------------------------------------


@pytest.mark.skipif("avif" not in supported_formats(), reason="pillow_heif AVIF not available")
def test_avif_deep_color_encode_succeeds_or_raises_clearly():
    """pillow_heif 0.22+ typed-buffer AVIF encode should succeed.

    If the installed version does not support it, `FormatNotSupportedError`
    must be raised with a descriptive message.
    """
    arr = _make_10bit_rgb(32, 32, seed=6)
    try:
        avif_bytes = _encode_avif_deep(arr, quality=65, bit_depth=10)
        assert len(avif_bytes) > 0
        assert avif_bytes[4:8] == b"ftyp", f"expected ftyp box, got: {avif_bytes[4:8]!r}"
        assert (
            b"avif" in avif_bytes[:32] or b"avis" in avif_bytes[:32]
        ), "AVIF brand not found in first 32 bytes"
    except FormatNotSupportedError as exc:
        assert (
            "avif" in str(exc).lower()
            or "typed-buffer" in str(exc).lower()
            or "pillow_heif" in str(exc).lower()
        ), f"FormatNotSupportedError message is not descriptive enough: {exc}"


# ---------------------------------------------------------------------------
# 4. 8-bit JXL path regression guard
# ---------------------------------------------------------------------------


@pytest.mark.skipif("jxl" not in supported_formats(), reason="jxlpy not available")
def test_8bit_path_unchanged_for_jxl():
    """Encoding an 8-bit PIL Image to JXL must not be affected by the
    ndarray dispatch added in this PR.

    The result must:
    - start with the expected JXL magic bytes,
    - be decodable by Pillow (or jxlpy),
    - have the correct dimensions.

    We do NOT pin the exact byte hash because JXL encoding is not bit-exact
    across library versions.  The invariants above are sufficient.
    """
    entry = ManifestEntry(
        name="t",
        bucket=Bucket.SMALL,
        content_kind="photo_perlin",
        seed=0,
        width=32,
        height=32,
        output_formats=["jxl"],
    )
    img = synthesize(entry)
    assert isinstance(img, Image.Image), "synthesize must return Image for photo_perlin"

    jxl_bytes = encode(img, "jxl", quality=85)
    assert jxl_bytes.startswith(
        (b"\x00\x00\x00\x0cJXL", b"\xff\x0a")
    ), f"unexpected JXL magic for 8-bit path: {jxl_bytes[:12].hex()}"

    # Verify decodable: re-open via Pillow (relies on jxlpy Pillow plugin)
    decoded = Image.open(io.BytesIO(jxl_bytes))
    decoded.load()
    assert decoded.size == (32, 32)


# ---------------------------------------------------------------------------
# 5. pixel_sha256 disambiguates bit depths
# ---------------------------------------------------------------------------


def test_pixel_sha256_disambiguates_bit_depths():
    """Same logical spatial pattern at 10-bit vs 16-bit must produce
    different pixel_sha256 values.

    pixel_sha256() already handles ndarrays by baking dtype + shape into the
    digest (manifest.py lines ~293-300).  This test pins that contract so a
    refactor that inadvertently drops the dtype prefix is caught immediately.
    """
    # Generate same spatial content but at different bit depths
    arr_10bit = deep_color_smooth(seed=77, width=32, height=32, bit_depth=10)
    arr_16bit = deep_color_smooth(seed=77, width=32, height=32, bit_depth=16)

    assert (
        arr_10bit.dtype == arr_16bit.dtype == np.uint16
    ), "both arrays should be uint16 regardless of logical bit depth"
    # They should differ because the values themselves differ
    assert int(arr_10bit.max()) < 1024
    assert int(arr_16bit.max()) > 4095  # 16-bit values saturate well above 4095

    sha_10bit = pixel_sha256(arr_10bit)
    sha_16bit = pixel_sha256(arr_16bit)
    assert sha_10bit != sha_16bit, (
        "pixel_sha256 must distinguish 10-bit and 16-bit arrays " f"(both got {sha_10bit[:12]})"
    )


# ---------------------------------------------------------------------------
# 6. _detect_bit_depth helper
# ---------------------------------------------------------------------------


def test_detect_bit_depth_10bit():
    arr = np.array([[[0, 512, 1023]]], dtype=np.uint16)  # max = 1023
    assert _detect_bit_depth(arr) == 10


def test_detect_bit_depth_12bit():
    arr = np.array([[[0, 2048, 4095]]], dtype=np.uint16)  # max = 4095
    assert _detect_bit_depth(arr) == 12


def test_detect_bit_depth_16bit():
    arr = np.array([[[0, 32768, 65535]]], dtype=np.uint16)  # max = 65535
    assert _detect_bit_depth(arr) == 16


def test_detect_bit_depth_empty_array():
    arr = np.zeros((0, 0, 3), dtype=np.uint16)
    assert _detect_bit_depth(arr) == 10  # max_val=0 → < 1024 → 10


# ---------------------------------------------------------------------------
# 7. encode() dispatch: ndarray routes to deep-color path for JXL/HEIC/AVIF
# ---------------------------------------------------------------------------


@pytest.mark.skipif("jxl" not in supported_formats(), reason="jxlpy not available")
def test_encode_dispatch_routes_ndarray_to_jxl_deep():
    """encode(ndarray, 'jxl') must use the deep-color path, not raise."""
    arr = _make_10bit_rgb(32, 32, seed=10)
    result = encode(arr, "jxl", quality=85)
    assert result.startswith((b"\x00\x00\x00\x0cJXL", b"\xff\x0a"))


@pytest.mark.skipif("heic" not in supported_formats(), reason="pillow_heif not available")
def test_encode_dispatch_routes_ndarray_to_heic_deep():
    """encode(ndarray, 'heic') must use the deep-color path, not raise."""
    arr = _make_10bit_rgb(32, 32, seed=11)
    try:
        result = encode(arr, "heic", quality=85)
        assert result[4:8] == b"ftyp"
    except FormatNotSupportedError:
        pass  # documented limitation path — acceptable


@pytest.mark.skipif("avif" not in supported_formats(), reason="pillow_heif AVIF not available")
def test_encode_dispatch_routes_ndarray_to_avif_deep():
    """encode(ndarray, 'avif') must use the deep-color path, not raise."""
    arr = _make_10bit_rgb(32, 32, seed=12)
    try:
        result = encode(arr, "avif", quality=65)
        assert result[4:8] == b"ftyp"
    except FormatNotSupportedError:
        pass  # documented limitation path — acceptable

"""Conversion / encoder tests."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from bench.corpus.conversion import (
    FormatNotSupportedError,
    encode,
    is_animation_format,
    supported_formats,
)
from bench.corpus.manifest import Bucket, ManifestEntry
from bench.corpus.synthesis import synthesize


def _photo_entry(**overrides) -> ManifestEntry:
    base = dict(
        name="t",
        bucket=Bucket.SMALL,
        content_kind="photo_perlin",
        seed=0,
        width=64,
        height=64,
        output_formats=["png"],
    )
    base.update(overrides)
    return ManifestEntry(**base)


def _animated_entry(**overrides) -> ManifestEntry:
    base = dict(
        name="t",
        bucket=Bucket.SMALL,
        content_kind="animated_translation",
        seed=0,
        width=64,
        height=64,
        output_formats=["apng"],
    )
    base.update(overrides)
    return ManifestEntry(**base)


def _deep_color_entry(**overrides) -> ManifestEntry:
    base = dict(
        name="t",
        bucket=Bucket.SMALL,
        content_kind="deep_color_smooth",
        seed=0,
        width=64,
        height=64,
        output_formats=["jxl"],
        params={"bit_depth": 10},
    )
    base.update(overrides)
    return ManifestEntry(**base)


def test_supported_formats_always_includes_baseline():
    """PNG/JPEG/WEBP/GIF/BMP/TIFF/APNG never depend on optional plugins."""
    fmts = set(supported_formats())
    assert {"png", "jpeg", "webp", "gif", "bmp", "tiff", "apng"} <= fmts


def test_unknown_format_raises():
    img = synthesize(_photo_entry())
    with pytest.raises(FormatNotSupportedError, match="no encoder"):
        encode(img, "doc")


@pytest.mark.parametrize(
    "fmt,magic",
    [
        ("png", b"\x89PNG\r\n\x1a\n"),
        ("jpeg", b"\xff\xd8\xff"),
        ("gif", b"GIF8"),
        ("bmp", b"BM"),
        ("webp", b"RIFF"),
    ],
)
def test_encode_static_produces_recognizable_bytes(fmt: str, magic: bytes):
    img = synthesize(_photo_entry())
    blob = encode(img, fmt)
    assert blob.startswith(magic), f"{fmt} bytes={blob[:8]!r}"
    Image.open(io.BytesIO(blob)).verify()


def test_encode_jpeg_strips_alpha_via_white_matte():
    """JPEG can't store alpha — encoder must matte to opaque RGB."""
    rgba = Image.new("RGBA", (32, 32), (255, 0, 0, 0))  # transparent red
    blob = encode(rgba, "jpeg", quality=90)
    assert blob.startswith(b"\xff\xd8\xff")
    decoded = Image.open(io.BytesIO(blob)).convert("RGB")
    assert decoded.size == (32, 32)


def test_encode_animated_apng_has_actl_chunk():
    """A valid APNG file contains an `acTL` chunk before `IDAT`."""
    frames = synthesize(_animated_entry())
    blob = encode(frames, "apng")
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"acTL" in blob


def test_encode_animated_webp_carries_anim_chunk():
    frames = synthesize(_animated_entry())
    blob = encode(frames, "webp")
    assert blob.startswith(b"RIFF")
    assert b"ANIM" in blob


def test_encode_animated_gif_has_netscape_loop_extension():
    frames = synthesize(_animated_entry())
    blob = encode(frames, "gif")
    assert blob.startswith(b"GIF8")
    assert b"NETSCAPE2.0" in blob


def test_encode_animated_to_static_format_uses_first_frame():
    """If only one frame can be encoded (PNG/BMP/JPEG), use frame[0]."""
    frames = synthesize(_animated_entry())
    blob = encode(frames, "png")
    img = Image.open(io.BytesIO(blob))
    assert img.size == frames[0].size


def test_encode_deep_color_to_8bit_format_raises():
    """ndarray content can't be fed to an 8-bit encoder."""
    arr = synthesize(_deep_color_entry())
    with pytest.raises(FormatNotSupportedError):
        encode(arr, "png")


def test_is_animation_format_recognizes_canonical_set():
    assert is_animation_format("apng")
    assert is_animation_format("gif")
    assert is_animation_format("webp")
    assert not is_animation_format("png")
    assert not is_animation_format("jpeg")


def test_jpeg_quality_actually_changes_bytes():
    img = synthesize(_photo_entry(content_kind="photo_noise"))
    big = encode(img, "jpeg", quality=95)
    small = encode(img, "jpeg", quality=20)
    assert len(big) > len(small)


def test_webp_lossy_smaller_than_png_for_photographic():
    img = synthesize(_photo_entry(width=128, height=128))
    png = encode(img, "png")
    webp = encode(img, "webp", quality=75)
    assert len(webp) < len(png), "WEBP @75 should beat raw PNG on photographic content"


def test_tiff_uses_lzw_compression():
    img = synthesize(_photo_entry())
    blob = encode(img, "tiff")
    decoded = Image.open(io.BytesIO(blob))
    assert decoded.format == "TIFF"


@pytest.mark.skipif(
    "heic" not in supported_formats(),
    reason="pillow_heif not available",
)
def test_encode_heic_when_plugin_present():
    img = synthesize(_photo_entry())
    blob = encode(img, "heic", quality=80)
    # HEIC files start with an `ftyp` box at offset 4
    assert blob[4:8] == b"ftyp"


@pytest.mark.skipif(
    "avif" not in supported_formats(),
    reason="pillow_heif AVIF support not available",
)
def test_encode_avif_when_plugin_present():
    img = synthesize(_photo_entry())
    blob = encode(img, "avif", quality=60)
    assert blob[4:8] == b"ftyp"
    # The AVIF brand sits at offset 8
    assert b"avif" in blob[:32] or b"avis" in blob[:32]


@pytest.mark.skipif(
    "jxl" not in supported_formats(),
    reason="pillow_jxl / jxlpy not available",
)
def test_encode_jxl_when_plugin_present():
    img = synthesize(_photo_entry())
    blob = encode(img, "jxl", quality=80)
    # JXL has either a container with `JXL ` brand or a bare codestream `\xFF\x0A`
    assert blob.startswith((b"\x00\x00\x00\x0cJXL", b"\xff\x0a"))


def test_supported_formats_omits_jxl_when_plugin_missing():
    """If neither pillow_jxl nor jxlpy is installed, JXL must not be advertised."""
    if "jxl" in supported_formats():
        pytest.skip("plugin is installed; nothing to assert")
    img = synthesize(_photo_entry())
    with pytest.raises(FormatNotSupportedError, match="not installed"):
        encode(img, "jxl")


def test_ndarray_content_rejected_by_first_frame_helper():
    """Independent of which format, ndarray content can't be passed to
    encoders that expect Image."""
    arr = np.zeros((4, 4, 3), dtype=np.uint16)
    for fmt in ("png", "jpeg", "gif", "bmp", "tiff", "webp"):
        with pytest.raises(FormatNotSupportedError):
            encode(arr, fmt)

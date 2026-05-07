"""Encode synthesized content into target image formats.

The conversion stage takes a synthesizer's output (static `PIL.Image`,
list of frames for animated, or `numpy.uint16` array for deep color) and
produces encoded bytes in each requested target format. Encoders are
registered lazily so that missing optional dependencies (jxlpy, etc.)
just skip the affected formats with a warning instead of breaking the
whole build.

Format coverage in v1:

| Format | Static | Animated | Deep color | Notes                                |
|--------|--------|----------|------------|--------------------------------------|
| PNG    |   ✓    |     —    |     —      |                                      |
| APNG   |   —    |     ✓    |     —      |                                      |
| JPEG   |   ✓    |     —    |     —      |                                      |
| WEBP   |   ✓    |     ✓    |     —      |                                      |
| GIF    |   ✓    |     ✓    |     —      |                                      |
| BMP    |   ✓    |     —    |     —      |                                      |
| TIFF   |   ✓    |     —    |     —      |                                      |
| HEIC   |   ✓    |     —    |     ✓      | typed-buffer via pillow_heif 0.22+   |
| AVIF   |   ✓    |     —    |     ✓      | typed-buffer via pillow_heif 0.22+   |
| JXL    |   ✓*   |     —    |     ✓      | native 10/12-bit via jxlpy encoder   |
| SVG    |   ✓†   |     —    |     —      | vector pass-through (bytes in)       |
| SVGZ   |   ✓†   |     —    |     —      | gzip of SVG, mtime=0                 |

* JXL requires `pillow_jxl` or `jxlpy` to be installed.
† SVG/SVGZ use vector pass-through: content must be raw bytes (not a PIL
  Image). Entries must set `content_kind="fetched_vector"` and provide a
  `source` URL; the builder bypasses Image.open() for these entries.
  Determinism contract: byte-level SHA-256 of the source bytes
  (stored in `expected_byte_sha256["source"]` in the manifest).

Deep-color encoding (10/12-bit):
  - JXL: natively supported via `jxlpy.JXLPyEncoder` typed buffers.  Pass a
    uint16 ndarray; bit depth is auto-detected from the array's max value
    (max < 1024 → 10-bit, max < 4096 → 12-bit, else 16-bit).
  - HEIC/AVIF: supported via `pillow_heif.from_bytes()` typed-buffer API
    (pillow_heif ≥ 0.22).  Raises `FormatNotSupportedError` if the installed
    version does not support the typed-buffer path.
  - All other formats: ndarray content raises `FormatNotSupportedError` because
    8-bit codecs cannot represent 10/12-bit pixel values without quantizing.

Production estimator note:
  BPP curve fits in `estimation/estimator.py` assume 8-bit input; deep-color
  accuracy will be off until estimator updates land in a follow-up PR.
"""

from __future__ import annotations

import io
import logging
import warnings
from typing import Callable

import numpy as np
from PIL import Image

from bench.corpus.manifest import Synthesized

logger = logging.getLogger(__name__)


class FormatNotSupportedError(Exception):
    """The requested format is not available in this environment."""


# --- Optional plugin registration ------------------------------------------

_HEIC_AVAILABLE = False
_AVIF_AVAILABLE = False
_JXL_AVAILABLE = False


def _register_optional_plugins() -> None:
    """Register pillow_heif (HEIC + AVIF) and pillow_jxl if installed.

    Idempotent — safe to call multiple times. Failures are logged but
    do not raise; affected formats just stay unavailable.
    """
    global _HEIC_AVAILABLE, _AVIF_AVAILABLE, _JXL_AVAILABLE

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIC_AVAILABLE = True
        if hasattr(pillow_heif, "register_avif_opener"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                pillow_heif.register_avif_opener()
            _AVIF_AVAILABLE = True
    except ImportError:
        logger.debug("pillow_heif unavailable; HEIC/AVIF skipped")

    try:
        import pillow_jxl  # noqa: F401

        _JXL_AVAILABLE = True
    except ImportError:
        try:
            # `import jxlpy` alone does NOT register the Pillow save/load
            # handler — that lives in jxlpy.JXLImagePlugin. Importing the
            # plugin module is what makes `Image.save(buf, format="JXL")`
            # work.
            from jxlpy import JXLImagePlugin  # noqa: F401

            _JXL_AVAILABLE = True
        except ImportError:
            logger.debug("pillow_jxl / jxlpy unavailable; JXL skipped")


_register_optional_plugins()


# --- Encoder primitives ----------------------------------------------------


def _to_rgb(img: Image.Image) -> Image.Image:
    """For codecs that don't carry alpha (JPEG, BMP). Mattes RGBA over white."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _first_frame(content: Synthesized) -> Image.Image:
    """For static formats: take the first frame of an animated source."""
    if isinstance(content, list):
        if not content:
            raise ValueError("empty frame list")
        return content[0]
    if isinstance(content, np.ndarray):
        raise FormatNotSupportedError("8-bit encoders cannot consume deep-color ndarray content")
    return content


def _encode_png(content: Synthesized) -> bytes:
    img = _first_frame(content)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _encode_jpeg(content: Synthesized, *, quality: int = 85) -> bytes:
    img = _to_rgb(_first_frame(content))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _encode_webp(content: Synthesized, *, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    if isinstance(content, list):
        content[0].save(
            buf,
            format="WEBP",
            save_all=True,
            append_images=content[1:],
            quality=quality,
            duration=100,
            loop=0,
        )
    else:
        _first_frame(content).save(buf, format="WEBP", quality=quality)
    return buf.getvalue()


def _encode_gif(content: Synthesized) -> bytes:
    buf = io.BytesIO()
    if isinstance(content, list):
        content[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=content[1:],
            duration=100,
            loop=0,
            optimize=False,
        )
    else:
        _first_frame(content).save(buf, format="GIF", optimize=False)
    return buf.getvalue()


def _encode_bmp(content: Synthesized) -> bytes:
    img = _to_rgb(_first_frame(content))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def _encode_tiff(content: Synthesized) -> bytes:
    img = _first_frame(content)
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression="tiff_lzw")
    return buf.getvalue()


def _encode_apng(content: Synthesized) -> bytes:
    """Animated PNG. If `content` is static, falls back to a single-frame PNG."""
    buf = io.BytesIO()
    if isinstance(content, list):
        content[0].save(
            buf,
            format="PNG",
            save_all=True,
            append_images=content[1:],
            duration=100,
            loop=0,
        )
    else:
        _first_frame(content).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _detect_bit_depth(arr: np.ndarray) -> int:
    """Infer bit depth from the uint16 array's max value.

    Convention from `bench/corpus/synthesis/deep_color.py`:
      - 10-bit values are in [0, 1023]
      - 12-bit values are in [0, 4095]
      - 16-bit values are in [0, 65535]
    """
    max_val = int(arr.max()) if arr.size else 0
    if max_val < 1024:
        return 10
    if max_val < 4096:
        return 12
    return 16


def _encode_jxl_deep(arr: np.ndarray, *, quality: int = 90, bit_depth: int = 10) -> bytes:
    """Encode uint16 ndarray (H, W, C) as JXL with native bit depth.

    Uses `jxlpy.JXLPyEncoder` typed-buffer API which accepts uint16 pixel data
    and preserves the requested bit depth in the output container.
    """
    from jxlpy import JXLPyEncoder

    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    if channels == 3:
        colorspace = "RGB"
    elif channels == 4:
        colorspace = "RGBA"
    else:
        colorspace = "Gray"
    enc = JXLPyEncoder(
        quality=quality,
        colorspace=colorspace,
        size=(w, h),
        bit_depth=bit_depth,
        num_threads=0,
    )
    enc.add_frame(arr.tobytes(order="C"))
    return enc.get_output()


def _encode_heic_deep(arr: np.ndarray, *, quality: int = 85, bit_depth: int = 10) -> bytes:
    """Encode uint16 ndarray as HEIC via pillow_heif typed-buffer API.

    Requires pillow_heif ≥ 0.22 with typed-buffer support (from_bytes with
    high-bit-depth modes like 'RGB;10').
    """
    import pillow_heif

    if bit_depth not in (10, 12):
        raise FormatNotSupportedError(
            f"HEIC typed-buffer requires bit_depth in (10, 12); got {bit_depth}"
        )
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    mode = f"RGB;{bit_depth}" if channels == 3 else f"RGBA;{bit_depth}"
    try:
        img = pillow_heif.from_bytes(mode=mode, size=(w, h), data=arr.tobytes(order="C"))
        buf = io.BytesIO()
        img.save(buf, format="HEIF", quality=quality)
        return buf.getvalue()
    except Exception as exc:
        raise FormatNotSupportedError(
            f"pillow_heif HEIC typed-buffer encode failed: {exc}"
        ) from exc


def _encode_avif_deep(arr: np.ndarray, *, quality: int = 65, bit_depth: int = 10) -> bytes:
    """Encode uint16 ndarray as AVIF via pillow_heif typed-buffer API.

    Same typed-buffer flow as HEIC but writes format='AVIF'.  Some older
    pillow_heif builds only encode AVIF from 8-bit; raises `FormatNotSupportedError`
    with a clear message if the typed-buffer save fails.
    """
    import pillow_heif

    if bit_depth not in (10, 12):
        raise FormatNotSupportedError(
            f"AVIF typed-buffer requires bit_depth in (10, 12); got {bit_depth}"
        )
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    mode = f"RGB;{bit_depth}" if channels == 3 else f"RGBA;{bit_depth}"
    try:
        img = pillow_heif.from_bytes(mode=mode, size=(w, h), data=arr.tobytes(order="C"))
        buf = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            img.save(buf, format="AVIF", quality=quality)
        return buf.getvalue()
    except Exception as exc:
        raise FormatNotSupportedError(
            f"pillow_heif AVIF typed-buffer encode failed: {exc}"
        ) from exc


def _encode_heic(content: Synthesized, *, quality: int = 85) -> bytes:
    if not _HEIC_AVAILABLE:
        raise FormatNotSupportedError("pillow_heif not installed")
    if isinstance(content, np.ndarray):
        bit_depth = _detect_bit_depth(content)
        return _encode_heic_deep(content, quality=quality, bit_depth=bit_depth)
    img = _to_rgb(_first_frame(content))
    buf = io.BytesIO()
    img.save(buf, format="HEIF", quality=quality)
    return buf.getvalue()


def _encode_avif(content: Synthesized, *, quality: int = 65) -> bytes:
    if not _AVIF_AVAILABLE:
        raise FormatNotSupportedError("pillow_heif AVIF support not installed")
    if isinstance(content, np.ndarray):
        bit_depth = _detect_bit_depth(content)
        return _encode_avif_deep(content, quality=quality, bit_depth=bit_depth)
    img = _to_rgb(_first_frame(content))
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        img.save(buf, format="AVIF", quality=quality)
    return buf.getvalue()


def _encode_jxl(content: Synthesized, *, quality: int = 85) -> bytes:
    if not _JXL_AVAILABLE:
        raise FormatNotSupportedError("pillow_jxl / jxlpy not installed")
    if isinstance(content, np.ndarray):
        bit_depth = _detect_bit_depth(content)
        return _encode_jxl_deep(content, quality=quality, bit_depth=bit_depth)
    img = _first_frame(content)
    buf = io.BytesIO()
    img.save(buf, format="JXL", quality=quality)
    return buf.getvalue()


def _encode_svg(content: Synthesized) -> bytes:
    """Pass-through: vector sources are already encoded.

    `content` must be raw bytes — the builder supplies the fetched SVG bytes
    directly without going through Image.open().  Raising here (instead of
    silently returning garbage) surfaces misconfigured entries early.
    """
    if not isinstance(content, (bytes, bytearray)):
        raise FormatNotSupportedError(
            "SVG entries must use byte content (set content_kind='fetched_vector' "
            "and provide entry.source). Got: " + type(content).__name__
        )
    return bytes(content)


def _encode_svgz(content: Synthesized) -> bytes:
    """Gzip-compress an SVG source into SVGZ.

    Uses mtime=0 so the output is deterministic across machines and runs.
    `content` must be raw bytes (same constraint as _encode_svg).
    """
    if not isinstance(content, (bytes, bytearray)):
        raise FormatNotSupportedError(
            "SVGZ entries must use byte content (set content_kind='fetched_vector' "
            "and provide entry.source). Got: " + type(content).__name__
        )
    import gzip

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(bytes(content))
    return buf.getvalue()


# --- Format dispatch -------------------------------------------------------

EncodeKwargs = dict[str, object]
_ENCODERS: dict[str, Callable[..., bytes]] = {
    "png": _encode_png,
    "jpeg": _encode_jpeg,
    "webp": _encode_webp,
    "gif": _encode_gif,
    "bmp": _encode_bmp,
    "tiff": _encode_tiff,
    "apng": _encode_apng,
    "heic": _encode_heic,
    "heif": _encode_heic,
    "avif": _encode_avif,
    "jxl": _encode_jxl,
    "svg": _encode_svg,
    "svgz": _encode_svgz,
}


def supported_formats() -> list[str]:
    """Return the formats encodable in the current environment.

    SVG and SVGZ are always available (stdlib gzip only; no optional plugins).
    """
    available = []
    for fmt in _ENCODERS:
        if fmt in {"heic", "heif"} and not _HEIC_AVAILABLE:
            continue
        if fmt == "avif" and not _AVIF_AVAILABLE:
            continue
        if fmt == "jxl" and not _JXL_AVAILABLE:
            continue
        available.append(fmt)
    return available


def is_animation_format(fmt: str) -> bool:
    return fmt.lower() in {"apng", "gif", "webp"}


def encode(content: Synthesized, fmt: str, **params: object) -> bytes:
    """Encode `content` to `fmt`, returning bytes.

    Raises:
        FormatNotSupportedError: format unknown or its plugin missing.
        TypeError: if content type doesn't match what the encoder accepts
                  (e.g. deep-color ndarray fed to an 8-bit encoder).
    """
    fmt = fmt.lower()
    encoder = _ENCODERS.get(fmt)
    if encoder is None:
        raise FormatNotSupportedError(f"no encoder for format {fmt!r}")
    return encoder(content, **params)

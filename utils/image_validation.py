"""Pre-optimization image validation: decompressed size and frame count limits.

Pillow opens images lazily — Image.open() reads headers without decompressing
pixel data. This module peeks at dimensions and rejects images that would consume
too much memory when fully decompressed, before any optimizer runs.
"""

import io

from PIL import Image

from exceptions import ImageTooLargeError

# Bytes per pixel by PIL mode (used to estimate decompressed size)
_BPP: dict[str, int] = {
    "1": 1,
    "L": 1,
    "P": 1,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "YCbCr": 3,
    "LAB": 3,
    "I": 4,
    "F": 4,
    "LA": 2,
    "PA": 2,
    "RGBa": 4,
    "I;16": 2,
}

MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024  # 512MB
MAX_FRAME_COUNT = 500


def validate_image_dimensions(data: bytes) -> None:
    """Reject images whose decompressed pixel data would exceed memory limits.

    Checks:
    1. Frame count for animated images (max 500 frames)
    2. Total decompressed size: width * height * bytes_per_pixel * frames

    SVG/SVGZ files are text-based and fail Image.open() — the except handler
    lets them through since they don't decompress to pixel data.
    DecompressionBombError is re-raised as ImageTooLargeError (not swallowed).
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            bpp = _BPP.get(img.mode, 4)  # Default to 4 (RGBA) if unknown mode
            n_frames = getattr(img, "n_frames", 1)
    except Image.DecompressionBombError:
        raise ImageTooLargeError(
            "Image dimensions exceed maximum allowed pixel count",
        )
    except Exception:
        return  # Not a raster image or invalid — let format detection handle it

    if n_frames > MAX_FRAME_COUNT:
        raise ImageTooLargeError(
            f"Animated image has {n_frames} frames, maximum is {MAX_FRAME_COUNT}",
            frames=n_frames,
            limit=MAX_FRAME_COUNT,
        )

    decompressed = width * height * bpp * n_frames

    if decompressed > MAX_DECOMPRESSED_BYTES:
        raise ImageTooLargeError(
            f"Decompressed image size ({decompressed // (1024 * 1024)}MB) "
            f"exceeds limit ({MAX_DECOMPRESSED_BYTES // (1024 * 1024)}MB)",
            decompressed_size=decompressed,
            limit=MAX_DECOMPRESSED_BYTES,
        )

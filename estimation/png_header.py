"""Pure-bytes PNG header parser.

Reads the PNG signature and IHDR chunk only — no Pillow, no I/O.
The caller supplies raw bytes (typically the first 33 bytes of the file).

Layout
------
Bytes 0-7    PNG signature: 89 50 4E 47 0D 0A 1A 0A
Bytes 8-11   IHDR chunk length (big-endian uint32, must equal 13)
Bytes 12-15  IHDR chunk type: b"IHDR"
Bytes 16-19  Image width (big-endian uint32)
Bytes 20-23  Image height (big-endian uint32)
Byte  24     Bit depth
Byte  25     Color type
Byte  26     Compression method (always 0)
Byte  27     Filter method (always 0)
Byte  28     Interlace method
Bytes 29-32  CRC (not verified here — caller trusts the file is well-formed)

Total: 33 bytes minimum for a parseable PNG header.

All integer reads use ``int.from_bytes(..., "big")`` (PNG is big-endian).
``memoryview`` is used for zero-copy slicing.
"""

from __future__ import annotations

from dataclasses import dataclass

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"  # 8 bytes
_IHDR_TYPE = b"IHDR"
_VALID_COLOR_TYPES = frozenset({0, 2, 3, 4, 6})
_VALID_BIT_DEPTHS = frozenset({1, 2, 4, 8, 16})
_MAX_DIMENSION = 2**31  # exclusive upper bound


@dataclass(frozen=True, slots=True)
class PngHeader:
    """Parsed PNG IHDR fields.

    Attributes
    ----------
    width, height : int
        Image dimensions in pixels. Both are > 0 and < 2^31.
    bit_depth : int
        One of {1, 2, 4, 8, 16}.
    color_type : int
        One of {0=L, 2=RGB, 3=P, 4=LA, 6=RGBA}.
    has_alpha : bool
        True when color_type is 4 (LA) or 6 (RGBA).
        Palette images (color_type 3) may have transparency via a tRNS chunk
        that is not present in the IHDR — this field reports *only* what IHDR
        encodes directly.
    is_palette : bool
        True when color_type == 3.
    """

    width: int
    height: int
    bit_depth: int
    color_type: int
    has_alpha: bool
    is_palette: bool


def parse_png_header(data: bytes) -> PngHeader | None:
    """Parse the PNG signature and IHDR chunk from *data*.

    Returns a ``PngHeader`` on success, or ``None`` on any structural failure.
    Never raises on malformed input.

    Only the first 33 bytes are inspected. *data* may be longer — the tail is
    ignored.

    Parameters
    ----------
    data : bytes
        Raw bytes starting at offset 0 of the PNG file (or a larger buffer).

    Returns
    -------
    PngHeader | None
        Parsed header, or ``None`` if the data is structurally invalid.
    """
    if len(data) < 33:
        return None

    mv = memoryview(data)

    # --- Signature (bytes 0-7) ---
    if bytes(mv[:8]) != _PNG_SIGNATURE:
        return None

    # --- IHDR chunk length (bytes 8-11) ---
    ihdr_length = int.from_bytes(mv[8:12], "big")
    if ihdr_length != 13:
        return None

    # --- IHDR chunk type (bytes 12-15) ---
    if bytes(mv[12:16]) != _IHDR_TYPE:
        return None

    # --- IHDR data (bytes 16-28) ---
    width = int.from_bytes(mv[16:20], "big")
    height = int.from_bytes(mv[20:24], "big")
    bit_depth = mv[24]
    color_type = mv[25]
    # bytes 26, 27, 28 are compression, filter, interlace — not needed

    # --- Validation ---
    if width == 0 or width >= _MAX_DIMENSION:
        return None
    if height == 0 or height >= _MAX_DIMENSION:
        return None
    if color_type not in _VALID_COLOR_TYPES:
        return None
    if bit_depth not in _VALID_BIT_DEPTHS:
        return None

    has_alpha = color_type in {4, 6}
    is_palette = color_type == 3

    return PngHeader(
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_type=color_type,
        has_alpha=has_alpha,
        is_palette=is_palette,
    )

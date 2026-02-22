import gzip
import struct
from enum import Enum

from exceptions import UnsupportedFormatError


class ImageFormat(str, Enum):
    PNG = "png"
    APNG = "apng"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"
    SVG = "svg"
    SVGZ = "svgz"
    AVIF = "avif"
    HEIC = "heic"
    TIFF = "tiff"
    BMP = "bmp"
    JXL = "jxl"


# MIME type mapping
MIME_TYPES = {
    ImageFormat.PNG: "image/png",
    ImageFormat.APNG: "image/apng",
    ImageFormat.JPEG: "image/jpeg",
    ImageFormat.WEBP: "image/webp",
    ImageFormat.GIF: "image/gif",
    ImageFormat.SVG: "image/svg+xml",
    ImageFormat.SVGZ: "image/svg+xml",
    ImageFormat.AVIF: "image/avif",
    ImageFormat.HEIC: "image/heic",
    ImageFormat.TIFF: "image/tiff",
    ImageFormat.BMP: "image/bmp",
    ImageFormat.JXL: "image/jxl",
}


def detect_format(data: bytes) -> ImageFormat:
    """Detect image format from magic bytes.

    Never trusts file extensions or Content-Type headers.

    Args:
        data: Raw image bytes (at least first 32 bytes needed).

    Returns:
        ImageFormat enum value.

    Raises:
        UnsupportedFormatError: If no known format matches.
    """
    if len(data) < 4:
        raise UnsupportedFormatError("File too small to identify format")

    # JXL bare codestream: \xFF\x0A (must check before JPEG's \xFF\xD8\xFF)
    if data[:2] == b"\xff\x0a":
        return ImageFormat.JXL

    # JXL ISOBMFF container: \x00\x00\x00\x0C\x4A\x58\x4C\x20\x0D\x0A\x87\x0A
    if data[:12] == b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a":
        return ImageFormat.JXL

    # PNG: \x89PNG\r\n\x1a\n
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if is_apng(data):
            return ImageFormat.APNG
        return ImageFormat.PNG

    # JPEG: \xFF\xD8\xFF
    if data[:3] == b"\xff\xd8\xff":
        return ImageFormat.JPEG

    # GIF: GIF87a or GIF89a
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ImageFormat.GIF

    # WebP: RIFF....WEBP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return ImageFormat.WEBP

    # BMP: BM
    if data[:2] == b"BM":
        return ImageFormat.BMP

    # TIFF: II*\x00 (little-endian) or MM\x00* (big-endian)
    if data[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return ImageFormat.TIFF

    # AVIF / HEIC: ISO BMFF ftyp box
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return _detect_isobmff(data)

    # SVGZ: gzip header \x1f\x8b
    if data[:2] == b"\x1f\x8b":
        try:
            decompressed = gzip.decompress(data)
            if _is_svg_content(decompressed):
                return ImageFormat.SVGZ
        except (gzip.BadGzipFile, OSError):
            pass

    # SVG: text-based detection (<?xml or <svg, after stripping BOM/whitespace)
    if _is_svg_content(data):
        return ImageFormat.SVG

    raise UnsupportedFormatError(
        "Unrecognized file format",
        detected_bytes=data[:16].hex(),
    )


def is_apng(data: bytes) -> bool:
    """Check if PNG data contains an acTL (animation control) chunk.

    Scans PNG chunks after the signature for an acTL chunk.
    The acTL chunk must appear before the first IDAT chunk.

    Args:
        data: Raw PNG bytes.

    Returns:
        True if acTL chunk found (animated PNG), False otherwise.
    """
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return False

    offset = 8  # Skip PNG signature

    while offset + 8 <= len(data):
        # Each chunk: 4-byte length + 4-byte type + data + 4-byte CRC
        if offset + 4 > len(data):
            break
        chunk_length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]

        if chunk_type == b"acTL":
            return True

        if chunk_type == b"IDAT":
            # Data started, no acTL found before it
            return False

        # Skip to next chunk: length + type(4) + data + CRC(4)
        offset += 4 + 4 + chunk_length + 4

    return False


def _detect_isobmff(data: bytes) -> ImageFormat:
    """Detect AVIF vs HEIC from ISO BMFF ftyp box.

    The ftyp box structure:
    - Bytes 0-3: box size (uint32 big-endian)
    - Bytes 4-7: 'ftyp'
    - Bytes 8-11: major brand (4 ASCII chars)
    - Bytes 12-15: minor version
    - Bytes 16+: compatible brands (4 bytes each)
    """
    major_brand = data[8:12]

    # JXL ISOBMFF container
    if major_brand == b"jxl ":
        return ImageFormat.JXL

    # AVIF brands
    if major_brand in (b"avif", b"avis"):
        return ImageFormat.AVIF

    # HEIC brands
    if major_brand in (b"heic", b"heix", b"mif1"):
        return ImageFormat.HEIC

    # Check compatible brands list if major brand doesn't match
    box_size = struct.unpack(">I", data[:4])[0]
    # Clamp to available data
    box_end = min(box_size, len(data))
    offset = 16  # Skip size + ftyp + major_brand + minor_version

    while offset + 4 <= box_end:
        compat_brand = data[offset : offset + 4]
        if compat_brand == b"jxl ":
            return ImageFormat.JXL
        if compat_brand in (b"avif", b"avis"):
            return ImageFormat.AVIF
        if compat_brand in (b"heic", b"heix", b"mif1"):
            return ImageFormat.HEIC
        offset += 4

    raise UnsupportedFormatError(
        "ISO BMFF file with unrecognized brand",
        major_brand=major_brand.decode("ascii", errors="replace"),
    )


def _is_svg_content(data: bytes) -> bool:
    """Check if data looks like SVG content.

    Strips BOM and leading whitespace, then checks for <?xml or <svg.
    """
    # Strip UTF-8 BOM
    text = data
    if text[:3] == b"\xef\xbb\xbf":
        text = text[3:]

    # Strip leading whitespace
    stripped = text.lstrip()

    # Check for XML declaration or SVG root element
    lower = stripped[:256].lower()
    return lower.startswith(b"<?xml") or lower.startswith(b"<svg")

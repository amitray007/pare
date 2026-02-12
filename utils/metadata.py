import io
import struct

from PIL import Image

from utils.format_detect import ImageFormat

# EXIF tag IDs to preserve
_ORIENTATION_TAG = 0x0112
# EXIF IFD pointers to strip entirely
_GPS_IFD = 0x8825


def strip_metadata_selective(
    data: bytes,
    fmt: ImageFormat,
    preserve_orientation: bool = True,
    preserve_icc: bool = True,
) -> bytes:
    """Strip non-essential metadata while preserving critical fields.

    Preserves:
        - EXIF Orientation tag (prevents rotated images in browsers)
        - ICC Color Profile (prevents color degradation for product photography)

    Strips:
        - GPS / Location data (privacy)
        - Camera/Device info
        - XMP / IPTC editorial metadata
        - Embedded thumbnails
        - Comments
    """
    if fmt in (ImageFormat.JPEG,):
        return _strip_jpeg_metadata(data, preserve_orientation, preserve_icc)
    if fmt in (ImageFormat.PNG, ImageFormat.APNG):
        return _strip_png_metadata(data, preserve_icc)
    if fmt == ImageFormat.TIFF:
        return _strip_pillow_metadata(data, fmt, preserve_orientation, preserve_icc)
    # WebP, GIF, SVG, BMP — metadata is stripped during optimization
    return data


def _strip_jpeg_metadata(
    data: bytes,
    preserve_orientation: bool,
    preserve_icc: bool,
) -> bytes:
    """Strip JPEG metadata, preserving orientation and ICC profile."""
    img = Image.open(io.BytesIO(data))

    # Extract values to preserve
    orientation = None
    icc_profile = None

    if preserve_orientation:
        exif_data = img.getexif()
        orientation = exif_data.get(_ORIENTATION_TAG)

    if preserve_icc:
        icc_profile = img.info.get("icc_profile")

    # Re-save without metadata
    output = io.BytesIO()
    save_kwargs = {"format": "JPEG", "quality": "keep", "subsampling": "keep"}

    # Build minimal EXIF with only orientation
    if orientation is not None:
        exif = Image.Exif()
        exif[_ORIENTATION_TAG] = orientation
        save_kwargs["exif"] = exif.tobytes()

    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    img.save(output, **save_kwargs)
    return output.getvalue()


def _strip_png_metadata(data: bytes, preserve_icc: bool) -> bytes:
    """Strip PNG text chunks, preserve iCCP and pHYs.

    PNG chunks to preserve: IHDR, PLTE, tRNS, IDAT, IEND, iCCP, pHYs, acTL, fcTL, fdAT
    PNG chunks to strip: tEXt, iTXt, zTXt (metadata text)
    """
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return data

    # Chunks to strip
    strip_types = {b"tEXt", b"iTXt", b"zTXt"}
    if not preserve_icc:
        strip_types.add(b"iCCP")

    output = bytearray(data[:8])  # PNG signature
    offset = 8

    while offset + 8 <= len(data):
        chunk_length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        # Full chunk: length(4) + type(4) + data(chunk_length) + CRC(4)
        chunk_end = offset + 4 + 4 + chunk_length + 4

        if chunk_end > len(data):
            # Incomplete chunk — keep remaining data as-is
            output.extend(data[offset:])
            break

        if chunk_type not in strip_types:
            output.extend(data[offset:chunk_end])

        offset = chunk_end

    return bytes(output)


def _strip_pillow_metadata(
    data: bytes,
    fmt: ImageFormat,
    preserve_orientation: bool,
    preserve_icc: bool,
) -> bytes:
    """Generic Pillow-based metadata stripping for TIFF and similar formats."""
    img = Image.open(io.BytesIO(data))

    orientation = None
    icc_profile = None

    if preserve_orientation:
        exif_data = img.getexif()
        orientation = exif_data.get(_ORIENTATION_TAG)

    if preserve_icc:
        icc_profile = img.info.get("icc_profile")

    # Map ImageFormat to Pillow format string
    pillow_format = {
        ImageFormat.TIFF: "TIFF",
    }.get(fmt, fmt.value.upper())

    output = io.BytesIO()
    save_kwargs = {"format": pillow_format}

    if orientation is not None:
        exif = Image.Exif()
        exif[_ORIENTATION_TAG] = orientation
        save_kwargs["exif"] = exif.tobytes()

    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    img.save(output, **save_kwargs)
    return output.getvalue()

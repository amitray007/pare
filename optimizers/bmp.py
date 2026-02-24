import asyncio
import io
import struct

import pyvips
from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


def load_bmp(data: bytes) -> pyvips.Image:
    """Load BMP into pyvips via Pillow (libvips has no native BMP support)."""
    pil_img = Image.open(io.BytesIO(data))
    if pil_img.mode == "P":
        pil_img = pil_img.convert("RGBA" if "transparency" in pil_img.info else "RGB")
    elif pil_img.mode not in ("L", "RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")

    bands = {"L": 1, "RGB": 3, "RGBA": 4}[pil_img.mode]
    raw = pil_img.tobytes()
    return pyvips.Image.new_from_memory(raw, pil_img.width, pil_img.height, bands, "uchar")


def encode_bmp_24(img: pyvips.Image) -> bytes:
    """Encode a pyvips image as uncompressed 24-bit BMP."""
    if img.bands == 1:
        img = img.bandjoin([img, img])
    elif img.bands == 4:
        img = img[:3]

    width, height = img.width, img.height
    row_stride = width * 3
    padding = (4 - (row_stride % 4)) % 4
    padded_row = row_stride + padding
    pixel_size = padded_row * height

    raw = img.write_to_memory()
    pad = b"\x00" * padding

    rows = []
    for y in range(height - 1, -1, -1):
        off = y * row_stride
        row = bytearray(raw[off : off + row_stride])
        # RGB -> BGR
        for x in range(0, row_stride, 3):
            row[x], row[x + 2] = row[x + 2], row[x]
        rows.append(bytes(row))
        rows.append(pad)

    file_header = struct.pack("<2sIHHI", b"BM", 14 + 40 + pixel_size, 0, 0, 14 + 40)
    info_header = struct.pack(
        "<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_size, 2835, 2835, 0, 0
    )
    return file_header + info_header + b"".join(rows)


def encode_bmp_palette(img: pyvips.Image) -> bytes:
    """Encode a palette (indexed) pyvips image as 8-bit BMP.

    Expects a 1-band image with palette metadata (from PNG palette load).
    Falls back to 24-bit if no palette is available.
    """
    if img.bands != 1 or not img.get_typeof("palette"):
        return encode_bmp_24(img)

    width, height = img.width, img.height
    padding = (4 - (width % 4)) % 4
    padded_row = width + padding
    pixel_size = padded_row * height

    # Extract palette: pyvips stores it as a 3-band Nx1 image
    palette_img = img.get("palette")
    n_colors = palette_img.width
    palette_raw = palette_img.write_to_memory()

    # Build BGRA palette entries (4 bytes each, 256 max)
    palette_data = bytearray()
    for i in range(n_colors):
        off = i * 3
        r, g, b = palette_raw[off], palette_raw[off + 1], palette_raw[off + 2]
        palette_data += struct.pack("BBBB", b, g, r, 0)
    # Pad to 256 entries
    palette_data += b"\x00" * (1024 - len(palette_data))

    raw = img.write_to_memory()
    pad = b"\x00" * padding
    rows = []
    for y in range(height - 1, -1, -1):
        off = y * width
        rows.append(raw[off : off + width])
        rows.append(pad)

    data_offset = 14 + 40 + 1024
    file_size = data_offset + pixel_size
    file_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, data_offset)
    info_header = struct.pack(
        "<IiiHHIIiiII", 40, width, height, 1, 8, 0, pixel_size, 2835, 2835, n_colors, 0
    )
    return file_header + info_header + bytes(palette_data) + b"".join(rows)


class BmpOptimizer(BaseOptimizer):
    """BMP optimization — quality-aware compression tiers.

    LOW  (quality >= 70): Lossless 32->24 bit downconversion only.
    MEDIUM (quality 50-69): Palette quantization to 256 colors.
    HIGH (quality < 50): Palette quantization to 256 colors (+ future RLE8).

    Each tier tries its methods plus all gentler methods, picks the smallest.
    Uses Pillow for BMP reading (libvips has no native BMP support).
    """

    format = ImageFormat.BMP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, best_method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, best_method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = load_bmp(data)

        # Drop alpha if fully opaque
        if img.bands == 4 and img.interpretation == "srgb":
            alpha = img[3]
            if alpha.min() == 255:
                img = img[:3]

        best = data
        best_method = "none"

        # Tier 1 (all presets): lossless 24-bit re-encode
        candidate = encode_bmp_24(img)
        if len(candidate) < len(best):
            best = candidate
            best_method = "pyvips-bmp"

        # Tier 2 (quality < 70): palette quantization to 256 colors
        if config.quality < 70:
            try:
                png_buf = img.pngsave_buffer(palette=True, Q=config.quality, effort=1)
                palette_img = pyvips.Image.new_from_buffer(png_buf, "")
                candidate = encode_bmp_palette(palette_img)
                if len(candidate) < len(best):
                    best = candidate
                    best_method = "pyvips-bmp-palette"
            except Exception:
                pass

        return best, best_method

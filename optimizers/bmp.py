import asyncio
import io
import logging
import struct

import pyvips
from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat

logger = logging.getLogger(__name__)


def load_bmp(data: bytes) -> pyvips.Image:
    """Load BMP into pyvips via Pillow (libvips has no native BMP support)."""
    pil_img = Image.open(io.BytesIO(data))
    if pil_img.mode == "P":
        pil_img = pil_img.convert("RGBA" if "transparency" in pil_img.info else "RGB")
    elif pil_img.mode not in ("L", "RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")

    interp = {"L": "b-w", "RGB": "srgb", "RGBA": "srgb"}[pil_img.mode]
    bands = {"L": 1, "RGB": 3, "RGBA": 4}[pil_img.mode]
    raw = pil_img.tobytes()
    img = pyvips.Image.new_from_memory(raw, pil_img.width, pil_img.height, bands, "uchar")
    return img.copy(interpretation=interp)


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
        # pyvips quantizes via libimagequant, Pillow writes the palette BMP
        # (pyvips decodes palette PNGs to RGB, losing the palette)
        if config.quality < 70:
            try:
                max_colors = 64 if config.quality < 50 else 256
                png_buf = img.pngsave_buffer(
                    palette=True, Q=config.quality, colours=max_colors, effort=1
                )
                pil_img = Image.open(io.BytesIO(png_buf))
                bio = io.BytesIO()
                pil_img.save(bio, format="BMP")
                candidate = bio.getvalue()
                if len(candidate) < len(best):
                    best = candidate
                    best_method = "pyvips-bmp-palette"
            except Exception:
                logger.warning("BMP palette quantization failed", exc_info=True)

        return best, best_method

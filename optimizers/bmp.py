import asyncio
import io
import struct

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class BmpOptimizer(BaseOptimizer):
    """BMP optimization — quality-aware compression tiers.

    LOW  (quality >= 70): Lossless 32→24 bit downconversion only (~0-25%).
    MEDIUM (quality 50-69): Palette quantization to 256 colors (~66%).
    HIGH (quality < 50): Palette quantization + RLE8 compression (66-99%).

    Each tier tries its methods plus all gentler methods, picks the smallest.
    """

    format = ImageFormat.BMP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, best_method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, best_method)

    def _optimize_sync(
        self, data: bytes, config: OptimizationConfig
    ) -> tuple[bytes, str]:
        """CPU-bound Pillow work — runs in a thread to avoid blocking the event loop."""
        img = Image.open(io.BytesIO(data))

        if img.mode == "RGBA":
            alpha = img.getchannel("A")
            if alpha.getextrema() == (255, 255):
                img = img.convert("RGB")
        elif img.mode not in ("RGB", "L", "P"):
            img = img.convert("RGB")

        best = data
        best_method = "none"

        # --- Tier 1 (all presets): lossless 24-bit re-encode ---
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        candidate = buf.getvalue()
        if len(candidate) < len(best):
            best = candidate
            best_method = "pillow-bmp"

        # --- Tier 2 (quality < 70): palette quantization (8-bit, 256 colors) ---
        if config.quality < 70:
            palette_img = self._quantize_to_palette(img)

            buf = io.BytesIO()
            palette_img.save(buf, format="BMP")
            candidate = buf.getvalue()
            if len(candidate) < len(best):
                best = candidate
                best_method = "pillow-bmp-palette"

            # --- Tier 3 (quality < 50): palette + RLE8 ---
            if config.quality < 50:
                candidate = self._encode_rle8_bmp(palette_img)
                if candidate is not None and len(candidate) < len(best):
                    best = candidate
                    best_method = "bmp-rle8"

        return best, best_method

    @staticmethod
    def _quantize_to_palette(img: Image.Image) -> Image.Image:
        """Quantize an image to 256-color palette using median-cut + dithering."""
        if img.mode == "P":
            return img
        rgb = img.convert("RGB")
        return rgb.quantize(
            colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG
        )

    @staticmethod
    def _encode_rle8_bmp(palette_img: Image.Image) -> bytes | None:
        """Construct a valid BMP file with BI_RLE8 compression.

        Returns None if the image isn't in palette mode or has > 256 colors.
        """
        if palette_img.mode != "P":
            return None

        w, h = palette_img.size
        pixels = palette_img.load()

        # --- Encode RLE8 data (BMP stores rows bottom-to-top) ---
        rle_data = bytearray()
        for y in range(h - 1, -1, -1):
            row = bytes(pixels[x, y] for x in range(w))
            _rle8_encode_row(row, rle_data)
            rle_data.extend(b"\x00\x00")  # end-of-line

        rle_data.extend(b"\x00\x01")  # end-of-bitmap

        # --- Build palette (256 BGRA entries) ---
        raw_palette = palette_img.getpalette()  # RGB flat list
        if raw_palette is None:
            return None
        palette_bytes = bytearray(1024)
        for i in range(256):
            idx = i * 3
            if idx + 2 < len(raw_palette):
                r, g, b = raw_palette[idx], raw_palette[idx + 1], raw_palette[idx + 2]
            else:
                r, g, b = 0, 0, 0
            off = i * 4
            palette_bytes[off] = b  # blue
            palette_bytes[off + 1] = g  # green
            palette_bytes[off + 2] = r  # red
            palette_bytes[off + 3] = 0  # reserved

        # --- Build headers ---
        rle_size = len(rle_data)
        pixel_offset = 14 + 40 + 1024  # file header + info header + palette
        file_size = pixel_offset + rle_size

        # BITMAPFILEHEADER (14 bytes)
        file_header = struct.pack(
            "<2sIHHI",
            b"BM",
            file_size,
            0,
            0,
            pixel_offset,
        )

        # BITMAPINFOHEADER (40 bytes)
        info_header = struct.pack(
            "<IiiHHIIiiII",
            40,  # biSize
            w,  # biWidth
            h,  # biHeight (positive = bottom-up)
            1,  # biPlanes
            8,  # biBitCount
            1,  # biCompression = BI_RLE8
            rle_size,  # biSizeImage
            0,  # biXPelsPerMeter
            0,  # biYPelsPerMeter
            256,  # biClrUsed
            0,  # biClrImportant
        )

        return file_header + info_header + bytes(palette_bytes) + bytes(rle_data)


def _rle8_encode_row(row: bytes, out: bytearray) -> None:
    """RLE8-encode a single row of pixel indices into *out*.

    Uses encoded runs for repeats and absolute mode for non-repeating
    sequences (count >= 3). Short non-repeating runs (1-2) are emitted
    as encoded runs of length 1 or 2 for simplicity.
    """
    n = len(row)
    i = 0

    while i < n:
        # Count consecutive identical bytes
        val = row[i]
        run = 1
        while i + run < n and row[i + run] == val and run < 255:
            run += 1

        if run >= 3:
            # Encoded run: [count, value]
            out.extend(bytes([run, val]))
            i += run
        else:
            # Collect non-repeating literal sequence
            lit_start = i
            i += run
            while i < n:
                # Peek ahead: if next is a run of 3+, stop literal
                val2 = row[i]
                peek = 1
                while i + peek < n and row[i + peek] == val2 and peek < 3:
                    peek += 1
                if peek >= 3:
                    break
                i += 1
                if i - lit_start >= 255:
                    break

            lit_len = i - lit_start
            if lit_len >= 3:
                # Absolute mode: [0x00, count, data...] padded to even
                out.append(0x00)
                out.append(lit_len)
                out.extend(row[lit_start : lit_start + lit_len])
                if lit_len % 2 != 0:
                    out.append(0x00)  # pad to even
            else:
                # Too short for absolute mode — emit as encoded runs
                for j in range(lit_start, lit_start + lit_len):
                    out.extend(bytes([1, row[j]]))

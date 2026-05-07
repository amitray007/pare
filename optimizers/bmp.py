import asyncio
import io
import struct

import numpy as np
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

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
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

        # --- Tier 1.5 (all presets): lossless palette for images with <= 256 colors ---
        if img.mode in ("RGB", "L"):
            lossless_result = self._try_lossless_palette(img)
            if lossless_result is not None:
                palette_img_lossless, palette_bmp, palette_method = lossless_result
                if len(palette_bmp) < len(best):
                    best = palette_bmp
                    best_method = palette_method
                # Also try RLE8 on the lossless palette image
                rle_candidate = self._encode_rle8_bmp(palette_img_lossless)
                if rle_candidate is not None and len(rle_candidate) < len(best):
                    best = rle_candidate
                    best_method = "bmp-rle8-lossless"

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
    def _try_lossless_palette(img: Image.Image) -> tuple[Image.Image, bytes, str] | None:
        """Try lossless conversion to 8-bit palette BMP.

        If the image has <= 256 unique colors, builds an exact palette
        (no quantization, no color loss) and returns (palette_img, bmp_bytes, method).
        Returns None if the image has too many colors.

        Uses numpy unique() for a single C-level pass over all pixels, which is
        dramatically faster than Python-level dict iteration for large images.
        """
        arr = np.asarray(img)  # H x W x 3 (RGB) or H x W (L)

        if arr.ndim == 3:
            # Pack RGB channels into a single uint32 for fast unique comparison.
            # Each pixel becomes (R << 16 | G << 8 | B) — unique per distinct color.
            packed = (
                arr[..., 0].astype(np.uint32) << 16
                | arr[..., 1].astype(np.uint32) << 8
                | arr[..., 2].astype(np.uint32)
            )
            is_rgb = True
        else:
            packed = arr.astype(np.uint32)
            is_rgb = False

        flat = packed.ravel()

        # Early-exit sample: if a small slice already has >256 unique values,
        # the full image won't fit in an 8-bit palette. This avoids a full-image
        # np.unique pass for photographic input that will be discarded anyway.
        SAMPLE_SIZE = 4096
        if flat.size > SAMPLE_SIZE:
            sample = flat[:SAMPLE_SIZE]
            if np.unique(sample).size > 256:
                return None

        unique_vals, inverse = np.unique(flat, return_inverse=True)

        if len(unique_vals) > 256:
            return None  # Too many colors for an 8-bit palette

        w, h = img.size

        # inverse gives the per-pixel index into unique_vals (in sorted order).
        pixel_indices = inverse.astype(np.uint8).tobytes()

        # Build palette image with exact color mapping
        palette_img = Image.new("P", (w, h))
        palette_img.frombytes(pixel_indices)

        # Build RGB palette (Pillow expects flat R,G,B list of 768 entries)
        flat_palette = [0] * 768
        for i, packed_val in enumerate(unique_vals):
            v = int(packed_val)
            if is_rgb:
                flat_palette[i * 3] = (v >> 16) & 0xFF  # R
                flat_palette[i * 3 + 1] = (v >> 8) & 0xFF  # G
                flat_palette[i * 3 + 2] = v & 0xFF  # B
            else:
                # Grayscale: replicate the single channel to R, G, B
                flat_palette[i * 3] = v
                flat_palette[i * 3 + 1] = v
                flat_palette[i * 3 + 2] = v
        palette_img.putpalette(flat_palette)

        buf = io.BytesIO()
        palette_img.save(buf, format="BMP")
        return palette_img, buf.getvalue(), "bmp-palette-lossless"

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
        arr = np.asarray(palette_img)  # H x W, dtype uint8 — no copy

        # --- Encode RLE8 data (BMP stores rows bottom-to-top) ---
        rle_data = bytearray()
        for y in range(h - 1, -1, -1):
            row = arr[y].tobytes()  # Contiguous C-level row extraction
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

    Uses numpy to detect run boundaries in a single C-level pass, then
    walks the pre-computed segment list. For rows shorter than 65 bytes
    (uncommon in practice) the pure-Python fallback is used because numpy
    setup overhead dominates at that scale.

    Encoding rules match BMP BI_RLE8 exactly:
    - Runs >= 3 identical bytes  -> encoded run: [count, value], capped at 255
    - Short sequences (run < 3)  -> accumulate into a literal block up to 255 bytes
        - Literal >= 3 bytes     -> absolute mode: [0x00, count, data, pad-to-even]
        - Literal < 3 bytes      -> individual encoded runs of length 1
    """
    n = len(row)
    if n == 0:
        return

    # For very short rows the numpy call overhead outweighs the savings.
    if n < 65:
        _rle8_encode_row_python(row, out, n)
        return

    arr = np.frombuffer(row, dtype=np.uint8)

    # Single C-level pass: locate every position where the value changes.
    change_pos = np.where(np.diff(arr) != 0)[0] + 1
    num_segs = len(change_pos) + 1

    # Build segment start indices and lengths as numpy arrays, then convert
    # to Python lists for the tight output-assembly loop (list indexing is
    # faster than numpy scalar extraction inside a Python for-loop).
    seg_starts = np.empty(num_segs, dtype=np.int32)
    seg_starts[0] = 0
    if num_segs > 1:
        seg_starts[1:] = change_pos

    seg_lengths_np = np.empty(num_segs, dtype=np.int32)
    if num_segs > 1:
        seg_lengths_np[:-1] = np.diff(seg_starts)
    seg_lengths_np[-1] = n - int(seg_starts[-1])

    seg_starts_list = seg_starts.tolist()
    seg_lengths_list = seg_lengths_np.tolist()
    vals_list = arr[seg_starts].tolist()  # pixel value for each segment

    seg_idx = 0
    pos = 0  # current byte position in *row*, mirrors 'i' in the Python version

    while seg_idx < num_segs:
        seg_len = seg_lengths_list[seg_idx]
        val = vals_list[seg_idx]

        if seg_len >= 3:
            # --- Encoded run mode, chunked to BMP max of 255 ---
            remaining = seg_len
            while remaining > 0:
                chunk = min(remaining, 255)
                out.append(chunk)
                out.append(val)
                remaining -= chunk
            pos += seg_len
            seg_idx += 1
        else:
            # --- Literal accumulation: gather short-run segments ---
            # Replicates the original byte-by-byte peek logic: accumulate
            # until hitting a run of >= 3 or reaching 255 bytes.  A run-of-2
            # segment that straddles the 255-byte boundary is split, matching
            # the original's single-byte advance behaviour.
            lit_start = pos
            pos += seg_len
            seg_idx += 1

            while seg_idx < num_segs:
                next_len = seg_lengths_list[seg_idx]
                if next_len >= 3:
                    break  # next is a long run — end the literal block
                available = 255 - (pos - lit_start)
                if available <= 0:
                    break
                if next_len <= available:
                    pos = seg_starts_list[seg_idx] + next_len
                    seg_idx += 1
                else:
                    # Partial segment: take only 'available' bytes, leave the rest.
                    # Mutate the list entry so the remainder is processed next time.
                    split_pos = seg_starts_list[seg_idx] + available
                    pos = split_pos
                    seg_starts_list[seg_idx] = split_pos
                    seg_lengths_list[seg_idx] = next_len - available
                    break

            lit_len = pos - lit_start
            if lit_len >= 3:
                # Absolute mode: [0x00, count, data, optional pad]
                out.append(0x00)
                out.append(lit_len)
                out.extend(row[lit_start : lit_start + lit_len])
                if lit_len % 2 != 0:
                    out.append(0x00)
            else:
                # Too short for absolute mode — individual encoded runs
                for j in range(lit_start, lit_start + lit_len):
                    out.append(1)
                    out.append(row[j])


def _rle8_encode_row_python(row: bytes, out: bytearray, n: int) -> None:
    """Pure-Python RLE8 row encoder — used for short rows (< 65 bytes).

    Identical logic to the original implementation; kept as a fallback because
    numpy call overhead dominates at small row widths.
    """
    i = 0
    while i < n:
        val = row[i]
        run = 1
        while i + run < n and row[i + run] == val and run < 255:
            run += 1

        if run >= 3:
            out.extend(bytes([run, val]))
            i += run
        else:
            lit_start = i
            i += run
            while i < n:
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
                out.append(0x00)
                out.append(lit_len)
                out.extend(row[lit_start : lit_start + lit_len])
                if lit_len % 2 != 0:
                    out.append(0x00)
            else:
                for j in range(lit_start, lit_start + lit_len):
                    out.extend(bytes([1, row[j]]))

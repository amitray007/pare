import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class BmpOptimizer(BaseOptimizer):
    """BMP optimization — quality-aware compression tiers.

    LOW  (quality >= 70): Lossless 32->24 bit downconversion only.
    MEDIUM (quality 50-69): Palette quantization to 256 colors.
    HIGH (quality < 50): Palette quantization to 256 colors (+ future RLE8).

    Each tier tries its methods plus all gentler methods, picks the smallest.
    """

    format = ImageFormat.BMP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, best_method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, best_method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")

        # Drop alpha if fully opaque
        if img.bands == 4 and img.interpretation == "srgb":
            alpha = img[3]
            if alpha.min() == 255:
                img = img[:3]

        best = data
        best_method = "none"

        # Tier 1 (all presets): lossless 24-bit re-encode
        candidate = img.write_to_buffer(".bmp")
        if len(candidate) < len(best):
            best = candidate
            best_method = "pyvips-bmp"

        # Tier 2 (quality < 70): palette quantization to 256 colors
        if config.quality < 70:
            # Quantize to 8-bit palette using pyvips
            # pyvips doesn't have direct BMP palette save, so we:
            # 1. Save as palette PNG (uses libimagequant)
            # 2. Re-load and save as BMP
            try:
                png_buf = img.pngsave_buffer(palette=True, Q=config.quality, effort=1)
                palette_img = pyvips.Image.new_from_buffer(png_buf, "")
                candidate = palette_img.write_to_buffer(".bmp")
                if len(candidate) < len(best):
                    best = candidate
                    best_method = "pyvips-bmp-palette"
            except Exception:
                pass

        return best, best_method

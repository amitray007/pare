import asyncio

import oxipng
import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat, is_apng


class PngOptimizer(BaseOptimizer):
    """PNG optimization: pyvips (libimagequant lossy) + oxipng (lossless enhancement).

    Pipeline:
    1. If APNG or lossless-only: pyvips lossless encode + oxipng enhancement
    2. Otherwise: pyvips lossy palette + oxipng enhancement, concurrently with lossless
    3. Pick smallest result

    Quality controls:
    - quality < 50: 64 max colors, effort=10 (aggressive)
    - quality < 70: 256 max colors, effort=7 (moderate)
    - quality >= 70: lossless only, oxipng level=2 (gentle)
    """

    format = ImageFormat.PNG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        animated = is_apng(data)
        if animated:
            self.format = ImageFormat.APNG

        strip = config.strip_metadata

        # APNG or lossless-only: skip lossy path
        if animated or not config.png_lossy:
            optimized = await asyncio.to_thread(self._lossless_encode, data, strip)
            # Enhancement: oxipng post-processing
            enhanced = await asyncio.to_thread(self._run_oxipng, optimized, config.quality)
            best = min([optimized, enhanced], key=len)
            return self._build_result(data, best, "oxipng")

        # Lossy + lossless paths run concurrently
        lossy_task = asyncio.to_thread(self._lossy_encode, data, config, strip)
        lossless_task = asyncio.to_thread(self._lossless_with_oxipng, data, config.quality, strip)

        lossy_result, lossless_result = await asyncio.gather(lossy_task, lossless_task)

        # Pick smallest
        candidates = []
        if lossy_result is not None:
            candidates.append((lossy_result, "pngquant + oxipng"))
        candidates.append((lossless_result, "oxipng"))

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)

    @staticmethod
    def _lossy_encode(data: bytes, config: OptimizationConfig, strip: bool) -> bytes | None:
        """Lossy PNG: pyvips palette quantization (libimagequant) + oxipng."""
        img = pyvips.Image.new_from_buffer(data, "")

        if config.quality < 50:
            colours = 64
            effort = 10
        else:
            colours = 256
            effort = 7

        try:
            lossy_buf = img.pngsave_buffer(
                palette=True,
                Q=config.quality,
                colours=colours,
                effort=effort,
                dither=1.0,
                strip=strip,
            )
        except Exception:
            return None

        # Post-process with oxipng
        oxipng_level = 4
        return oxipng.optimize_from_memory(lossy_buf, level=oxipng_level)

    @staticmethod
    def _lossless_encode(data: bytes, strip: bool) -> bytes:
        """Lossless PNG encode via pyvips."""
        img = pyvips.Image.new_from_buffer(data, "")
        return img.pngsave_buffer(compression=9, effort=10, strip=strip)

    @staticmethod
    def _lossless_with_oxipng(data: bytes, quality: int, strip: bool) -> bytes:
        """Lossless pyvips encode + oxipng enhancement."""
        img = pyvips.Image.new_from_buffer(data, "")
        lossless_buf = img.pngsave_buffer(compression=9, effort=10, strip=strip)

        oxipng_level = 4 if quality < 70 else 2
        return oxipng.optimize_from_memory(lossless_buf, level=oxipng_level)

    @staticmethod
    def _run_oxipng(data: bytes, quality: int) -> bytes:
        """Run oxipng for lossless post-processing enhancement."""
        oxipng_level = 4 if quality < 70 else 2
        return oxipng.optimize_from_memory(data, level=oxipng_level)

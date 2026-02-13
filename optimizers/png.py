import asyncio

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat, is_apng
from utils.metadata import strip_metadata_selective
from utils.subprocess_runner import run_tool


class PngOptimizer(BaseOptimizer):
    """PNG optimization: pngquant (lossy) + oxipng (lossless).

    Pipeline:
    1. If APNG → oxipng only (pngquant destroys animation frames)
    2. If png_lossy=False → oxipng only (user requested lossless)
    3. Otherwise → pngquant → oxipng on result
    4. pngquant exit code 99 (quality threshold not met) → fallback to oxipng on original

    Quality controls aggressiveness:
    - quality < 50:  64 max colors, floor=1, speed=1, oxipng level=6 (aggressive)
    - quality < 70:  256 max colors, floor=1, speed=4, oxipng level=4 (moderate)
    - quality >= 70: lossless only, oxipng level=2 (gentle)
    """

    format = ImageFormat.PNG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        animated = is_apng(data)
        if animated:
            self.format = ImageFormat.APNG

        # Strip metadata first (preserves iCCP, pHYs; strips tEXt chunks)
        if config.strip_metadata:
            data_clean = strip_metadata_selective(
                data,
                ImageFormat.APNG if animated else ImageFormat.PNG,
            )
        else:
            data_clean = data

        # Quality-dependent oxipng level: higher = slower but better compression
        # Level 6 = 180 filter trials (too slow for API use on large images)
        # Level 4 = 24 trials (good tradeoff for aggressive preset)
        # Level 3 is NOT used — it misses critical filters for screenshots
        if config.quality < 50:
            oxipng_level = 4
        elif config.quality < 70:
            oxipng_level = 4
        else:
            oxipng_level = 2

        # APNG or lossless-only: skip pngquant
        if animated or not config.png_lossy:
            optimized = await asyncio.to_thread(self._run_oxipng, data_clean, oxipng_level)
            method = "oxipng"
            return self._build_result(data, optimized, method)

        # Quality-dependent pngquant settings
        if config.quality < 50:
            max_colors = 64
            speed = 3  # good palette quality, 3-5x faster than speed=1
        else:
            max_colors = 256
            speed = 4  # default balanced

        # Lossy path: run pngquant and oxipng-baseline concurrently
        (pngquant_result, success), oxipng_only = await asyncio.gather(
            self._run_pngquant(data_clean, config.quality, max_colors, speed),
            asyncio.to_thread(self._run_oxipng, data_clean, oxipng_level),
        )

        if success and pngquant_result:
            # Squeeze extra bytes from the lossy result
            lossy_optimized = await asyncio.to_thread(
                self._run_oxipng, pngquant_result, oxipng_level
            )
            # Pick the smaller of lossy and lossless paths — pngquant can
            # produce a larger file when dithering inflates palette PNGs.
            use_lossy = len(lossy_optimized) <= len(oxipng_only)

            if use_lossy:
                optimized = lossy_optimized
                method = "pngquant + oxipng"
            else:
                optimized = oxipng_only
                method = "oxipng"
        else:
            # pngquant couldn't meet quality threshold — lossless only
            optimized = oxipng_only
            method = "oxipng"

        return self._build_result(data, optimized, method)

    async def _run_pngquant(
        self,
        data: bytes,
        quality: int,
        max_colors: int = 256,
        speed: int = 4,
    ) -> tuple[bytes | None, bool]:
        """Run pngquant with quality-dependent settings.

        Uses floor=1 so pngquant always succeeds (never exit 99).
        Max colors varies by quality: 128 for aggressive, 256 for moderate.
        Speed: 1=slowest/best palette, 4=default, 11=fastest/roughest.

        Returns:
            (output_bytes, success). success=False when exit code 99
            (quality threshold cannot be met).
        """
        cmd = [
            "pngquant",
            str(max_colors),
            "--quality",
            f"1-{quality}",
            "--speed",
            str(speed),
            "-",
            "--output",
            "-",
        ]

        stdout, stderr, returncode = await run_tool(
            cmd,
            data,
            allowed_exit_codes={99},
        )

        if returncode == 99:
            return None, False

        return stdout, True

    def _run_oxipng(self, data: bytes, level: int = 2) -> bytes:
        """Run oxipng in-process via pyoxipng library (no subprocess).

        Level: 0=fastest/least compression, 6=slowest/best compression.
        """
        import oxipng

        return oxipng.optimize_from_memory(data, level=level)

from exceptions import OptimizationError
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

        # APNG or lossless-only: skip pngquant
        if animated or not config.png_lossy:
            optimized = self._run_oxipng(data_clean)
            method = "oxipng"
            return self._build_result(data, optimized, method)

        # Lossy path: pngquant then oxipng
        pngquant_result, success = await self._run_pngquant(data_clean, config.quality)

        if success and pngquant_result:
            # Squeeze extra bytes from the lossy result
            optimized = self._run_oxipng(pngquant_result)
            method = "pngquant + oxipng"
        else:
            # pngquant couldn't meet quality threshold — lossless only
            optimized = self._run_oxipng(data_clean)
            method = "oxipng"

        return self._build_result(data, optimized, method)

    async def _run_pngquant(
        self, data: bytes, quality: int
    ) -> tuple[bytes | None, bool]:
        """Run pngquant with quality range.

        The quality floor is max(1, quality - 15) and ceiling is quality.
        Example: quality=80 → --quality 65-80

        Returns:
            (output_bytes, success). success=False when exit code 99
            (quality threshold cannot be met).
        """
        q_floor = max(1, quality - 15)
        q_ceil = quality

        stdout, stderr, returncode = await run_tool(
            ["pngquant", "--quality", f"{q_floor}-{q_ceil}", "-", "--output", "-"],
            data,
            allowed_exit_codes={99},
        )

        if returncode == 99:
            return None, False

        return stdout, True

    def _run_oxipng(self, data: bytes) -> bytes:
        """Run oxipng in-process via pyoxipng library (no subprocess)."""
        import oxipng

        return oxipng.optimize_from_memory(data)

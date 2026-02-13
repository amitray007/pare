import asyncio
import io

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.subprocess_runner import run_tool


class JpegOptimizer(BaseOptimizer):
    """JPEG optimization: MozJPEG cjpeg (lossy) + jpegtran (lossless).

    Pipeline:
    1. Always try lossy mozjpeg at target quality
    2. Always try lossless jpegtran (Huffman optimization)
    3. Pick smallest result
    4. Enforce optimization guarantee (output <= input)
    """

    format = ImageFormat.JPEG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        bmp_data = await asyncio.to_thread(self._decode_to_bmp, data, config.strip_metadata)

        # Run mozjpeg and jpegtran concurrently
        mozjpeg_out, jpegtran_out = await asyncio.gather(
            self._run_cjpeg(bmp_data, config.quality, config.progressive_jpeg),
            self._run_jpegtran(data, config.progressive_jpeg),
        )

        # Cap mozjpeg (lossy) if max_reduction is set.
        # Jpegtran (lossless) is never capped — no quality loss.
        if config.max_reduction is not None:
            moz_red = (1 - len(mozjpeg_out) / len(data)) * 100
            if moz_red > config.max_reduction:
                mozjpeg_out = await self._cap_mozjpeg(bmp_data, data, config)

        # Pick smallest between (possibly capped) mozjpeg and jpegtran
        candidates = [(mozjpeg_out, "mozjpeg"), (jpegtran_out, "jpegtran")]
        best_data, best_method = min(candidates, key=lambda x: len(x[0]))

        return self._build_result(data, best_data, best_method)

    async def _cap_mozjpeg(
        self,
        bmp_data: bytes,
        original: bytes,
        config: OptimizationConfig,
    ) -> bytes:
        """Binary search cjpeg quality to cap lossy reduction at max_reduction.

        Returns the cjpeg output at the lowest quality that stays within
        the cap, or the original bytes if even q=100 exceeds the cap.
        """
        target = config.max_reduction
        orig_size = len(original)

        out_100 = await self._run_cjpeg(bmp_data, 100, config.progressive_jpeg)
        red_100 = (1 - len(out_100) / orig_size) * 100
        if red_100 > target:
            return original  # mozjpeg can't help within cap

        lo, hi = config.quality, 100
        best_out = out_100

        for _ in range(5):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            out_mid = await self._run_cjpeg(bmp_data, mid, config.progressive_jpeg)
            red_mid = (1 - len(out_mid) / orig_size) * 100
            if red_mid > target:
                lo = mid
            else:
                hi = mid
                best_out = out_mid

        return best_out

    def _decode_to_bmp(self, data: bytes, strip_metadata: bool) -> bytes:
        """Decode JPEG to BMP format for cjpeg input.

        MozJPEG's cjpeg doesn't accept JPEG input — it needs
        BMP, PPM, or Targa. We decode via Pillow and output BMP.
        """
        img = Image.open(io.BytesIO(data))
        # Convert to RGB if RGBA (BMP for cjpeg should be RGB)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        output = io.BytesIO()
        img.save(output, format="BMP")
        return output.getvalue()

    async def _run_cjpeg(self, bmp_data: bytes, quality: int, progressive: bool) -> bytes:
        """Run MozJPEG cjpeg on BMP input."""
        cmd = ["cjpeg", "-quality", str(quality)]
        if progressive:
            cmd.append("-progressive")
        stdout, stderr, rc = await run_tool(cmd, bmp_data)
        return stdout

    async def _run_jpegtran(self, data: bytes, progressive: bool) -> bytes:
        """Run jpegtran for lossless Huffman table optimization."""
        cmd = ["jpegtran", "-optimize", "-copy", "none"]
        if progressive:
            cmd.append("-progressive")
        stdout, stderr, rc = await run_tool(cmd, data)
        return stdout

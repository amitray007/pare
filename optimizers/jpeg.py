import io

from PIL import Image

from estimation.header_analysis import estimate_jpeg_quality_from_qtable
from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.metadata import strip_metadata_selective
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
        # Always try lossy mozjpeg at target quality
        bmp_data = self._decode_to_bmp(data, config.strip_metadata)
        mozjpeg_out = await self._run_cjpeg(
            bmp_data, config.quality, config.progressive_jpeg
        )

        # Always try lossless jpegtran
        jpegtran_out = await self._run_jpegtran(data, config.progressive_jpeg)

        # Pick smallest
        candidates = [(mozjpeg_out, "mozjpeg"), (jpegtran_out, "jpegtran")]
        best_data, best_method = min(candidates, key=lambda x: len(x[0]))

        return self._build_result(data, best_data, best_method)

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

    async def _run_cjpeg(
        self, bmp_data: bytes, quality: int, progressive: bool
    ) -> bytes:
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

import io

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.metadata import strip_metadata_selective
from utils.subprocess_runner import run_tool


class JpegOptimizer(BaseOptimizer):
    """JPEG optimization: MozJPEG cjpeg (lossy) + jpegtran (lossless).

    Pipeline:
    1. Estimate input quality from quantization tables
    2. If input quality <= target → jpegtran only (lossless Huffman optimization)
    3. If input quality > target → decode to BMP, pipe to cjpeg for lossy re-encode
    4. Enforce optimization guarantee
    """

    format = ImageFormat.JPEG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        input_quality = self._estimate_jpeg_quality(data)

        if input_quality <= config.quality:
            # Already at or below target quality — lossless only
            optimized = await self._run_jpegtran(data, config.progressive_jpeg)
            method = "jpegtran"
        else:
            # Lossy re-encode via MozJPEG
            bmp_data = self._decode_to_bmp(data, config.strip_metadata)
            optimized = await self._run_cjpeg(
                bmp_data, config.quality, config.progressive_jpeg
            )
            method = "mozjpeg"

        return self._build_result(data, optimized, method)

    def _estimate_jpeg_quality(self, data: bytes) -> int:
        """Estimate input JPEG quality from quantization tables.

        Compares the image's quantization table values against
        standard JPEG quantization matrices to estimate quality.

        Returns:
            Estimated quality (1-100). Returns 100 if cannot determine.
        """
        try:
            img = Image.open(io.BytesIO(data))
            qtables = img.quantization
            if not qtables:
                return 100

            # Use the luminance table (table 0) to estimate quality.
            # Standard JPEG quality formula: the average quantization
            # value inversely correlates with quality.
            table = qtables[0] if 0 in qtables else list(qtables.values())[0]
            avg_q = sum(table) / len(table)

            # Approximate mapping: avg_q ~1 = q100, avg_q ~50 = q50, avg_q ~100 = q1
            # This is a rough heuristic — exact mapping depends on the encoder.
            if avg_q <= 1:
                return 100
            elif avg_q <= 2:
                return 95
            elif avg_q <= 5:
                return 90
            elif avg_q <= 10:
                return 80
            elif avg_q <= 20:
                return 70
            elif avg_q <= 40:
                return 60
            elif avg_q <= 60:
                return 50
            elif avg_q <= 80:
                return 40
            elif avg_q <= 100:
                return 30
            else:
                return 20
        except Exception:
            return 100  # Can't determine — assume high quality

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

import asyncio
import io
import os
import shutil
import tempfile

from PIL import Image

from optimizers.base import BaseOptimizer
from optimizers.utils import binary_search_quality
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.subprocess_runner import run_tool


class WebpOptimizer(BaseOptimizer):
    """WebP optimization: Pillow + cwebp CLI run concurrently, pick smallest.

    Pipeline:
    1. Run Pillow re-encode and cwebp CLI in parallel
    2. Pick the smallest result
    3. If max_reduction set and exceeded, binary search quality
    """

    format = ImageFormat.WEBP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        # Run Pillow and cwebp concurrently — pick smallest
        pillow_task = asyncio.to_thread(self._pillow_optimize, data, config.quality)
        cwebp_task = self._cwebp_fallback(data, config.quality)

        pillow_result, cwebp_result = await asyncio.gather(pillow_task, cwebp_task)

        best = pillow_result
        method = "pillow"
        if cwebp_result and len(cwebp_result) < len(best):
            best = cwebp_result
            method = "cwebp"

        # Cap reduction if max_reduction is set
        if config.max_reduction is not None:
            reduction = (1 - len(best) / len(data)) * 100
            if reduction > config.max_reduction:
                capped = await asyncio.to_thread(self._find_capped_quality, data, config)
                if capped is not None:
                    best = capped
                    method = "pillow"

        return self._build_result(data, best, method)

    def _find_capped_quality(self, data: bytes, config: OptimizationConfig) -> bytes | None:
        """Binary search Pillow quality to cap reduction at max_reduction."""

        def encode_fn(quality: int) -> bytes:
            return self._pillow_optimize(data, quality)

        return binary_search_quality(
            encode_fn, len(data), config.max_reduction, lo=config.quality, hi=100
        )

    def _pillow_optimize(self, data: bytes, quality: int) -> bytes:
        """In-process WebP optimization via Pillow.

        Handles both static and animated WebP.
        For animated: preserves all frames via save_all=True.
        """
        img = Image.open(io.BytesIO(data))
        output = io.BytesIO()

        is_animated = getattr(img, "n_frames", 1) > 1

        save_kwargs = {
            "format": "WEBP",
            "quality": quality,
            "method": 4,  # Good compression, 2-3x faster than method=6
        }

        if is_animated:
            save_kwargs["save_all"] = True
            save_kwargs["minimize_size"] = True

        img.save(output, **save_kwargs)
        return output.getvalue()

    async def _cwebp_fallback(self, data: bytes, quality: int) -> bytes | None:
        """Fallback to cwebp CLI.

        cwebp doesn't support stdin/stdout piping, so temp files are required.
        Returns None if cwebp is not available.
        """
        if not shutil.which("cwebp"):
            return None

        # cwebp requires file input/output
        try:
            with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as infile:
                infile.write(data)
                infile.flush()
                in_path = infile.name

            out_path = in_path + ".out.webp"

            stdout, stderr, rc = await run_tool(
                ["cwebp", "-q", str(quality), "-m", "4", "-mt", in_path, "-o", out_path],
                b"",  # No stdin needed
            )

            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    return f.read()
            return None
        except Exception:
            return None
        finally:
            for path in (in_path, out_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

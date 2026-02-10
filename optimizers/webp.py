import io
import shutil
import tempfile
import os

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.subprocess_runner import run_tool


class WebpOptimizer(BaseOptimizer):
    """WebP optimization: Pillow (primary) + cwebp CLI (fallback).

    Pipeline:
    1. Check for animated WebP (n_frames > 1) → Pillow with save_all
    2. Pillow decode + re-encode at target quality
    3. If Pillow result >= 90% of input → try cwebp fallback
    4. Use whichever output is smaller
    """

    format = ImageFormat.WEBP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        pillow_result = self._pillow_optimize(data, config.quality)

        # If Pillow result is suspiciously poor, try cwebp fallback
        if len(pillow_result) >= len(data) * 0.9:
            cwebp_result = await self._cwebp_fallback(data, config.quality)
            if cwebp_result and len(cwebp_result) < len(pillow_result):
                return self._build_result(data, cwebp_result, "cwebp")

        method = "pillow"
        return self._build_result(data, pillow_result, method)

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
            "method": 6,  # Slowest but best compression
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
            with tempfile.NamedTemporaryFile(
                suffix=".webp", delete=False
            ) as infile:
                infile.write(data)
                infile.flush()
                in_path = infile.name

            out_path = in_path + ".out.webp"

            stdout, stderr, rc = await run_tool(
                ["cwebp", "-q", str(quality), "-m", "6", in_path, "-o", out_path],
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

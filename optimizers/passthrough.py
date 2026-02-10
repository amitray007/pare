import io

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


# Pillow format strings for each passthrough format
_PILLOW_FORMATS = {
    ImageFormat.TIFF: "TIFF",
    ImageFormat.BMP: "BMP",
    ImageFormat.PSD: None,  # Pillow can read PSD but not write it
}


class PassthroughOptimizer(BaseOptimizer):
    """Best-effort optimization for TIFF, BMP, PSD via Pillow.

    These formats have limited optimization potential. The service
    attempts Pillow decode → re-encode with optimization flags.
    If no reduction is achieved, the original is returned unchanged.
    """

    def __init__(self, fmt: ImageFormat):
        self.format = fmt

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        pillow_fmt = _PILLOW_FORMATS.get(self.format)

        if pillow_fmt is None:
            # Can't write this format (e.g., PSD) — return original
            return self._build_result(data, data, "none")

        try:
            img = Image.open(io.BytesIO(data))
            output = io.BytesIO()

            save_kwargs = {"format": pillow_fmt}

            # TIFF-specific: use compression
            if self.format == ImageFormat.TIFF:
                save_kwargs["compression"] = "tiff_lzw"

            img.save(output, **save_kwargs)
            optimized = output.getvalue()
        except Exception:
            return self._build_result(data, data, "none")

        method = f"pillow-{self.format.value}"
        return self._build_result(data, optimized, method)

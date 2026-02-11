import io
import struct

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class BmpOptimizer(BaseOptimizer):
    """BMP optimization — downconvert 32-bit to 24-bit and re-encode.

    32-bit BMPs use 4 bytes per pixel (often with an unused alpha/padding
    channel). Pillow decodes to RGB and re-encodes as 24-bit, saving ~25%.
    For 24-bit BMPs, re-encoding normalizes padding and strips extra data.
    """

    format = ImageFormat.BMP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        img = Image.open(io.BytesIO(data))

        # Ensure RGB for re-encode (Pillow opens 32-bit BMP as RGB already,
        # but RGBA mode is possible with some BMP variants)
        if img.mode == "RGBA":
            alpha = img.getchannel("A")
            if alpha.getextrema() == (255, 255):
                img = img.convert("RGB")
        elif img.mode not in ("RGB", "L", "P"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="BMP")
        optimized = buf.getvalue()
        return self._build_result(data, optimized, "pillow-bmp")

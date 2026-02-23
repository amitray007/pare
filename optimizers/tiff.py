import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class TiffOptimizer(BaseOptimizer):
    """TIFF optimization — try multiple compression methods, pick smallest.

    Lossless methods (all presets): deflate, lzw
    Lossy method (quality < 70): jpeg at config.quality

    All methods run concurrently via asyncio.gather.
    """

    format = ImageFormat.TIFF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        img = pyvips.Image.new_from_buffer(data, "")

        strip = config.strip_metadata

        methods = [
            ("deflate", {"compression": "deflate", "strip": strip}),
            ("lzw", {"compression": "lzw", "strip": strip}),
        ]

        # Lossy JPEG-in-TIFF for quality < 70 and compatible band count
        if config.quality < 70 and img.bands in (1, 3):
            methods.append(
                ("tiff_jpeg", {"compression": "jpeg", "Q": config.quality, "strip": strip})
            )

        results = await asyncio.gather(
            *[
                asyncio.to_thread(self._try_compression, img, method_name, save_kwargs)
                for method_name, save_kwargs in methods
            ]
        )

        best, best_method = data, "none"
        for candidate, method in results:
            if candidate is not None and len(candidate) < len(best):
                best, best_method = candidate, method

        return self._build_result(data, best, best_method)

    @staticmethod
    def _try_compression(
        img: pyvips.Image, method_name: str, save_kwargs: dict
    ) -> tuple[bytes | None, str]:
        try:
            return img.tiffsave_buffer(**save_kwargs), method_name
        except Exception:
            return None, method_name

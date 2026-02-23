import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class AvifOptimizer(BaseOptimizer):
    """AVIF optimization — lossy re-encoding via pyvips (libheif + libaom).

    Quality mapping: avif_quality = max(30, min(90, quality + 10))
    """

    format = ImageFormat.AVIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")
        avif_quality = max(30, min(90, config.quality + 10))

        save_kwargs = {
            "Q": avif_quality,
            "compression": "av1",
            "effort": 4,  # 0=fastest, 9=slowest
            "strip": config.strip_metadata,
        }

        result = img.heifsave_buffer(**save_kwargs)
        return result, "avif-reencode"

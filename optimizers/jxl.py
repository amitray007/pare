import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class JxlOptimizer(BaseOptimizer):
    """JPEG XL optimization — lossy re-encoding via pyvips (libjxl).

    Quality mapping: jxl_quality = max(30, min(95, quality + 10))
    """

    format = ImageFormat.JXL

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")
        jxl_quality = max(30, min(95, config.quality + 10))

        save_kwargs = {
            "Q": jxl_quality,
            "effort": 7,  # 1=fastest, 9=slowest
            "strip": config.strip_metadata,
        }

        result = img.jxlsave_buffer(**save_kwargs)
        return result, "jxl-reencode"

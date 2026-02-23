import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class JpegOptimizer(BaseOptimizer):
    """JPEG optimization via pyvips (jpegli).

    Pipeline:
    1. Encode at target quality with optimize_coding=True (Huffman optimization)
    2. If max_reduction set and exceeded, binary search quality
    3. Enforce output-never-larger guarantee

    jpegli provides 35% better compression than mozjpeg at equivalent quality.
    optimize_coding=True replaces jpegtran's lossless Huffman optimization.
    """

    format = ImageFormat.JPEG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")

        save_kwargs = {
            "Q": config.quality,
            "optimize_coding": True,
            "strip": config.strip_metadata,
        }

        best = img.jpegsave_buffer(**save_kwargs)
        method = "jpegli"

        # Cap reduction if max_reduction is set
        if config.max_reduction is not None:
            reduction = (1 - len(best) / len(data)) * 100
            if reduction > config.max_reduction:
                capped = self._find_capped_quality(img, data, config)
                if capped is not None:
                    best = capped

        return best, method

    def _find_capped_quality(
        self,
        img: pyvips.Image,
        data: bytes,
        config: OptimizationConfig,
    ) -> bytes | None:
        target = config.max_reduction
        orig_size = len(data)

        out_100 = img.jpegsave_buffer(Q=100, optimize_coding=True, strip=config.strip_metadata)
        if (1 - len(out_100) / orig_size) * 100 > target:
            return None

        lo, hi = config.quality, 100
        best_out = out_100

        for _ in range(5):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            out_mid = img.jpegsave_buffer(
                Q=mid, optimize_coding=True, strip=config.strip_metadata
            )
            if (1 - len(out_mid) / orig_size) * 100 > target:
                lo = mid
            else:
                hi = mid
                best_out = out_mid

        return best_out

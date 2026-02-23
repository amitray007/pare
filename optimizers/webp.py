import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class WebpOptimizer(BaseOptimizer):
    """WebP optimization via pyvips (libwebp).

    Pipeline:
    1. Encode at target quality with effort=4
    2. If max_reduction set and exceeded, binary search quality
    3. Enforce output-never-larger guarantee
    """

    format = ImageFormat.WEBP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")

        # Check if animated (multi-page)
        n_pages = img.get("n-pages") if img.get_typeof("n-pages") else 1
        is_animated = n_pages > 1

        best = self._encode(img, config.quality, is_animated)
        method = "pyvips-webp"

        # Cap reduction if max_reduction is set
        if config.max_reduction is not None:
            reduction = (1 - len(best) / len(data)) * 100
            if reduction > config.max_reduction:
                capped = self._find_capped_quality(img, data, config, is_animated)
                if capped is not None:
                    best = capped

        return best, method

    @staticmethod
    def _encode(img: pyvips.Image, quality: int, animated: bool) -> bytes:
        save_kwargs = {"Q": quality, "effort": 4}
        if animated:
            save_kwargs["page_height"] = (
                img.get("page-height") if img.get_typeof("page-height") else img.height
            )
        return img.webpsave_buffer(**save_kwargs)

    def _find_capped_quality(
        self,
        img: pyvips.Image,
        data: bytes,
        config: OptimizationConfig,
        animated: bool,
    ) -> bytes | None:
        target = config.max_reduction
        orig_size = len(data)

        out_100 = self._encode(img, 100, animated)
        if (1 - len(out_100) / orig_size) * 100 > target:
            return None

        lo, hi = config.quality, 100
        best_out = out_100

        for _ in range(5):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            out_mid = self._encode(img, mid, animated)
            if (1 - len(out_mid) / orig_size) * 100 > target:
                lo = mid
            else:
                hi = mid
                best_out = out_mid

        return best_out

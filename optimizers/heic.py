import asyncio
import io

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class HeicOptimizer(BaseOptimizer):
    """HEIC lossless optimization — metadata stripping only.

    Identical approach to AVIF: no decode/re-encode to avoid
    generation loss. Only strips non-essential metadata.
    """

    format = ImageFormat.HEIC

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        if not config.strip_metadata:
            return self._build_result(data, data, "none")

        try:
            optimized = await asyncio.to_thread(self._strip_metadata, data)
        except Exception:
            return self._build_result(data, data, "none")

        return self._build_result(data, optimized, "metadata-strip")

    def _strip_metadata(self, data: bytes) -> bytes:
        """Strip metadata from HEIC using pillow-heif."""
        import pillow_heif

        heif_file = pillow_heif.open_heif(data)
        img = heif_file.to_pillow()

        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "HEIF", "quality": -1}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        if len(result) < len(data):
            return result
        return data

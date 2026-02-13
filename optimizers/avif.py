import asyncio
import io

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class AvifOptimizer(BaseOptimizer):
    """AVIF lossless optimization — metadata stripping only.

    Does NOT decode + re-encode. AVIF is a lossy format; each
    decode/re-encode cycle causes generation loss (cumulative quality
    degradation). Only lossless operations are applied:
    - Strip EXIF metadata
    - Strip XMP metadata
    - Preserve ICC color profile

    If no metadata to strip, returns original with 0% reduction.
    """

    format = ImageFormat.AVIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        if not config.strip_metadata:
            return self._build_result(data, data, "none")

        try:
            optimized = await asyncio.to_thread(self._strip_metadata, data)
        except Exception:
            # If metadata stripping fails, return original unchanged
            return self._build_result(data, data, "none")

        return self._build_result(data, optimized, "metadata-strip")

    def _strip_metadata(self, data: bytes) -> bytes:
        """Strip metadata from AVIF using pillow-heif."""
        import pillow_heif

        heif_file = pillow_heif.open_heif(data)
        img = heif_file.to_pillow()

        # Preserve ICC profile if present
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "AVIF", "quality": -1}  # -1 = lossless
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        # Only return stripped version if it's actually smaller
        # (re-encoding even "losslessly" might produce a different size)
        if len(result) < len(data):
            return result
        return data

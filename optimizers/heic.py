import asyncio
import io

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class HeicOptimizer(BaseOptimizer):
    """HEIC optimization — lossy re-encoding + metadata stripping.

    Same pattern as AVIF: try metadata strip and lossy re-encoding,
    pick the smallest result. Uses x265 (HEVC) via pillow-heif.

    Quality thresholds:
    - quality < 50 (HIGH):  HEIC q=50, aggressive re-encode
    - quality < 70 (MEDIUM): HEIC q=70, moderate re-encode
    - quality >= 70 (LOW):  HEIC q=90, conservative re-encode
    """

    format = ImageFormat.HEIC

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        candidates = []

        if config.strip_metadata:
            try:
                stripped = await asyncio.to_thread(self._strip_metadata, data)
                candidates.append((stripped, "metadata-strip"))
            except Exception:
                pass

        try:
            reencoded = await asyncio.to_thread(self._reencode, data, config.quality)
            candidates.append((reencoded, "heic-reencode"))
        except Exception:
            pass

        if not candidates:
            return self._build_result(data, data, "none")

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)

    def _strip_metadata(self, data: bytes) -> bytes:
        """Strip metadata from HEIC using pillow-heif."""
        import pillow_heif

        pillow_heif.register_heif_opener()
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

    def _reencode(self, data: bytes, quality: int) -> bytes:
        """Re-encode HEIC at target quality via x265 (HEVC) encoder."""
        import pillow_heif

        pillow_heif.register_heif_opener()
        heif_file = pillow_heif.open_heif(data)
        img = heif_file.to_pillow()
        icc_profile = img.info.get("icc_profile")

        heic_quality = max(30, min(90, quality + 10))

        output = io.BytesIO()
        save_kwargs = {
            "format": "HEIF",
            "quality": heic_quality,
        }
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        return output.getvalue()

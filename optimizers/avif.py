import asyncio
import io

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class AvifOptimizer(BaseOptimizer):
    """AVIF optimization — lossy re-encoding + metadata stripping.

    Pipeline:
    1. Always try metadata stripping (cheap, lossless)
    2. Try lossy re-encoding at target quality via AV1
    3. Pick smallest result
    4. Enforce optimization guarantee (output <= input)

    Quality thresholds:
    - quality < 50 (HIGH):  AVIF q=50, aggressive re-encode
    - quality < 70 (MEDIUM): AVIF q=70, moderate re-encode
    - quality >= 70 (LOW):  AVIF q=90, conservative re-encode

    Uses pillow-avif-plugin (libavif) for AVIF decode/encode.
    """

    format = ImageFormat.AVIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        candidates = []

        # Always try metadata strip (cheap, lossless)
        if config.strip_metadata:
            try:
                stripped = await asyncio.to_thread(self._strip_metadata, data)
                candidates.append((stripped, "metadata-strip"))
            except Exception:
                pass

        # Try lossy re-encoding at target quality
        try:
            reencoded = await asyncio.to_thread(self._reencode, data, config.quality)
            candidates.append((reencoded, "avif-reencode"))
        except Exception:
            pass

        if not candidates:
            return self._build_result(data, data, "none")

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)

    def _strip_metadata(self, data: bytes) -> bytes:
        """Strip metadata from AVIF — re-encode losslessly without metadata."""
        import pillow_avif  # noqa: F401 — registers AVIF plugin

        img = Image.open(io.BytesIO(data))
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "AVIF", "quality": 100}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        if len(result) < len(data):
            return result
        return data

    def _reencode(self, data: bytes, quality: int) -> bytes:
        """Re-encode AVIF at target quality via libavif (AV1 encoder)."""
        import pillow_avif  # noqa: F401 — registers AVIF plugin

        img = Image.open(io.BytesIO(data))
        icc_profile = img.info.get("icc_profile")

        # Map Pare quality (1-100, lower=aggressive) to AVIF quality
        avif_quality = max(30, min(90, quality + 10))

        output = io.BytesIO()
        save_kwargs = {
            "format": "AVIF",
            "quality": avif_quality,
            "speed": 6,  # 0=slowest/best, 10=fastest
        }
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        return output.getvalue()

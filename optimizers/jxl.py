import asyncio
import io

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class JxlOptimizer(BaseOptimizer):
    """JPEG XL optimization — lossy re-encoding + metadata stripping.

    Same pattern as AVIF/HEIC: try metadata strip and lossy re-encoding,
    pick the smallest result. Uses jxlpy (pillow-jxl-plugin) for encode/decode.

    Quality thresholds:
    - quality < 50 (HIGH):  JXL q=50, aggressive re-encode
    - quality < 70 (MEDIUM): JXL q=70, moderate re-encode
    - quality >= 70 (LOW):  JXL q=90, conservative re-encode
    """

    format = ImageFormat.JXL

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        tasks = []

        if config.strip_metadata:
            tasks.append(asyncio.to_thread(self._strip_metadata, data))
        tasks.append(asyncio.to_thread(self._reencode, data, config.quality))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        method_names = []
        if config.strip_metadata:
            method_names.append("metadata-strip")
        method_names.append("jxl-reencode")

        for result, method in zip(results, method_names):
            if not isinstance(result, Exception):
                candidates.append((result, method))

        if not candidates:
            return self._build_result(data, data, "none")

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)

    def _strip_metadata(self, data: bytes) -> bytes:
        """Strip metadata from JXL — re-encode losslessly without metadata."""
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            import jxlpy  # noqa: F401

        img = Image.open(io.BytesIO(data))
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "JXL", "lossless": True}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        if len(result) < len(data):
            return result
        return data

    def _reencode(self, data: bytes, quality: int) -> bytes:
        """Re-encode JXL at target quality."""
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            import jxlpy  # noqa: F401

        img = Image.open(io.BytesIO(data))
        icc_profile = img.info.get("icc_profile")

        # Map Pare quality (1-100, lower=aggressive) to JXL quality
        jxl_quality = max(30, min(95, quality + 10))

        output = io.BytesIO()
        save_kwargs = {
            "format": "JXL",
            "quality": jxl_quality,
        }
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        return output.getvalue()

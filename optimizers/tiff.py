import io

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.metadata import strip_metadata_selective


class TiffOptimizer(BaseOptimizer):
    """TIFF optimization — try multiple compression methods, pick smallest.

    Tries tiff_adobe_deflate (zlib-based, usually best for most content)
    and tiff_lzw, keeping whichever produces the smallest output.
    """

    format = ImageFormat.TIFF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        if config.strip_metadata:
            data = strip_metadata_selective(data, ImageFormat.TIFF)

        img = Image.open(io.BytesIO(data))

        # Preserve metadata fields for re-encoding
        exif_bytes = img.info.get("exif")
        icc_profile = img.info.get("icc_profile")

        methods = ["tiff_adobe_deflate", "tiff_lzw"]
        best = data  # default: keep original
        best_method = "none"

        for compression in methods:
            buf = io.BytesIO()
            save_kwargs = {"format": "TIFF", "compression": compression}

            if not config.strip_metadata:
                if exif_bytes:
                    save_kwargs["exif"] = exif_bytes
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile

            img.save(buf, **save_kwargs)
            candidate = buf.getvalue()
            if len(candidate) < len(best):
                best = candidate
                best_method = compression

        return self._build_result(data, best, best_method)

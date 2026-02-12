import io

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.metadata import strip_metadata_selective


class TiffOptimizer(BaseOptimizer):
    """TIFF optimization — try multiple compression methods, pick smallest.

    Lossless methods (all presets):
    - tiff_adobe_deflate (zlib-based, usually best for most content)
    - tiff_lzw

    Lossy method (quality < 70 only):
    - tiff_jpeg at config.quality (JPEG-in-TIFF, large savings on photos)

    Quality controls aggressiveness:
    - quality < 50:  lossy JPEG + lossless, pick smallest (aggressive)
    - quality < 70:  lossy JPEG + lossless, pick smallest (moderate)
    - quality >= 70: lossless only (gentle)
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

        # Lossy JPEG-in-TIFF for aggressive/moderate presets.
        # Only works for RGB/L modes (not RGBA/palette).
        use_lossy = config.quality < 70 and img.mode in ("RGB", "L")
        if use_lossy:
            methods.append("tiff_jpeg")

        best = data  # default: keep original
        best_method = "none"

        for compression in methods:
            buf = io.BytesIO()
            save_kwargs = {"format": "TIFF", "compression": compression}

            if compression == "tiff_jpeg":
                save_kwargs["quality"] = config.quality

            if not config.strip_metadata:
                if exif_bytes:
                    save_kwargs["exif"] = exif_bytes
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile

            try:
                img.save(buf, **save_kwargs)
            except Exception:
                continue
            candidate = buf.getvalue()
            if len(candidate) < len(best):
                best = candidate
                best_method = compression

        return self._build_result(data, best, best_method)

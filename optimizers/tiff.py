import asyncio
import io

from PIL import Image

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.metadata import strip_metadata_selective

# Images above this threshold run compression methods sequentially
# to avoid memory-expensive img.copy() calls. Below it, parallel is fine.
PARALLEL_PIXEL_THRESHOLD = 5_000_000  # 5 megapixels


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

    For images >5MP, methods run sequentially on the shared Image to avoid
    memory-expensive img.copy() calls. Below 5MP, methods run concurrently.
    """

    format = ImageFormat.TIFF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        if config.strip_metadata:
            data = strip_metadata_selective(data, ImageFormat.TIFF)

        img, exif_bytes, icc_profile = await asyncio.to_thread(self._decode, data)

        methods = ["tiff_adobe_deflate", "tiff_lzw"]
        if config.quality < 70 and img.mode in ("RGB", "L"):
            methods.append("tiff_jpeg")

        pixel_count = img.size[0] * img.size[1]

        if pixel_count < PARALLEL_PIXEL_THRESHOLD:
            # Small image: parallel with copies (fast, memory is negligible)
            results = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        self._try_compression,
                        img.copy(),
                        compression,
                        config,
                        exif_bytes,
                        icc_profile,
                    )
                    for compression in methods
                ]
            )
        else:
            # Large image: sequential on shared img (no copies, saves memory)
            results = []
            for compression in methods:
                result = await asyncio.to_thread(
                    self._try_compression, img, compression, config, exif_bytes, icc_profile
                )
                results.append(result)

        best, best_method = data, "none"
        for candidate, method in results:
            if candidate is not None and len(candidate) < len(best):
                best, best_method = candidate, method

        return self._build_result(data, best, best_method)

    def _decode(self, data: bytes) -> tuple[Image.Image, bytes | None, bytes | None]:
        """Decode TIFF and extract metadata — runs once, shared across threads."""
        img = Image.open(io.BytesIO(data))
        img.load()  # Force pixel data into memory for thread-safe concurrent saves
        exif_bytes = img.info.get("exif")
        icc_profile = img.info.get("icc_profile")
        return img, exif_bytes, icc_profile

    def _try_compression(
        self,
        img: Image.Image,
        compression: str,
        config: OptimizationConfig,
        exif_bytes: bytes | None,
        icc_profile: bytes | None,
    ) -> tuple[bytes | None, str]:
        """Try a single compression method — returns (bytes, method) or (None, method)."""
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
            return None, compression
        return buf.getvalue(), compression

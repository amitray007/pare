import io

from PIL import Image

from optimizers.pillow_reencode import PillowReencodeOptimizer
from utils.format_detect import ImageFormat


class AvifOptimizer(PillowReencodeOptimizer):
    """AVIF optimization — lossy re-encoding + metadata stripping.

    Uses pillow-avif-plugin (libavif) for AVIF decode/encode.

    Quality thresholds (via clamp_quality with offset=10, lo=30, hi=90):
    - quality < 50 (HIGH):  AVIF q=50, aggressive re-encode
    - quality < 70 (MEDIUM): AVIF q=70, moderate re-encode
    - quality >= 70 (LOW):  AVIF q=90, conservative re-encode
    """

    format = ImageFormat.AVIF
    pillow_format = "AVIF"
    strip_method_name = "metadata-strip"
    reencode_method_name = "avif-reencode"
    quality_min = 30
    quality_max = 90
    quality_offset = 10
    extra_save_kwargs = {"speed": 6}  # 0=slowest/best, 10=fastest

    def _ensure_plugin(self):
        import pillow_avif  # noqa: F401 — registers AVIF plugin

    def _strip_metadata_from_img(self, img: Image.Image, original_data: bytes) -> bytes:
        """AVIF strip uses quality=100 instead of lossless=True."""
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "AVIF", "quality": 100}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        return result if len(result) < len(original_data) else original_data

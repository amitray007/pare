import io

from PIL import Image

from optimizers.pillow_reencode import PillowReencodeOptimizer
from utils.format_detect import ImageFormat


class HeicOptimizer(PillowReencodeOptimizer):
    """HEIC optimization — lossy re-encoding + metadata stripping.

    Uses x265 (HEVC) via pillow-heif.

    Quality thresholds (via clamp_quality with offset=10, lo=30, hi=90):
    - quality < 50 (HIGH):  HEIC q=50, aggressive re-encode
    - quality < 70 (MEDIUM): HEIC q=70, moderate re-encode
    - quality >= 70 (LOW):  HEIC q=90, conservative re-encode
    """

    format = ImageFormat.HEIC
    pillow_format = "HEIF"
    strip_method_name = "metadata-strip"
    reencode_method_name = "heic-reencode"
    quality_min = 30
    quality_max = 90
    quality_offset = 10

    def _ensure_plugin(self):
        import pillow_heif

        pillow_heif.register_heif_opener()

    def _open_image(self, data: bytes) -> Image.Image:
        """HEIC uses pillow-heif's direct decoder for reliable loading."""
        import pillow_heif

        self._ensure_plugin()
        heif_file = pillow_heif.open_heif(data)
        return heif_file.to_pillow()

    def _strip_metadata_from_img(self, img: Image.Image, original_data: bytes) -> bytes:
        """HEIC strip uses quality=-1 (lossless) via pillow-heif."""
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "HEIF", "quality": -1}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        return result if len(result) < len(original_data) else original_data

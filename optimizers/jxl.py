try:
    import pillow_jxl  # noqa: F401 — registers JXL codec with Pillow
except ImportError:
    import jxlpy  # noqa: F401

from optimizers.pillow_reencode import PillowReencodeOptimizer
from utils.format_detect import ImageFormat


class JxlOptimizer(PillowReencodeOptimizer):
    """JPEG XL optimization — lossy re-encoding + metadata stripping.

    Uses jxlpy (pillow-jxl-plugin) for encode/decode.

    Quality thresholds (via clamp_quality with offset=10, lo=30, hi=95):
    - quality < 50 (HIGH):  JXL q=50, aggressive re-encode
    - quality < 70 (MEDIUM): JXL q=70, moderate re-encode
    - quality >= 70 (LOW):  JXL q=90, conservative re-encode
    """

    format = ImageFormat.JXL
    pillow_format = "JXL"
    strip_method_name = "metadata-strip"
    reencode_method_name = "jxl-reencode"
    quality_min = 30
    quality_max = 95  # JXL supports higher quality ceiling than AVIF/HEIC
    quality_offset = 10

    def _ensure_plugin(self):
        pass  # Plugin registered at module import time

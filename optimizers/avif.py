import io
import struct

from PIL import Image

from optimizers.pillow_reencode import PillowReencodeOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat

# Metadata box types whose byte count contributes to strippable overhead.
# "colr" boxes are intentionally excluded: the strip path preserves the ICC
# profile (icc_profile= kwarg) and nclx is baked into the bitstream — neither
# is actually removed, so counting them would over-inflate meta_bytes.
_METADATA_BOX_TYPES = frozenset({b"Exif", b"xml ", b"xmp ", b"mime"})

# ISOBMFF container types we recurse into (limited — avoids over-parsing).
_CONTAINER_TYPES = frozenset({b"meta", b"iprp", b"ipco"})


def _parse_avif_metadata(data: bytes) -> tuple[int, int, int]:
    """Extract image dimensions and strippable metadata size from AVIF box headers.

    Walks the top-level boxes and recurses one level into meta/iprp/ipco to find
    the first ispe box (width/height) and accumulate bytes from metadata boxes.
    No pixel data is decoded — only header bytes are read.

    Returns:
        (width, height, metadata_bytes) where width/height==0 means parse failed
        and the caller should fall back to the decode path.  Only truly strippable
        boxes (Exif, xml, xmp, mime) are counted; colr boxes are excluded for
        both kinds (kind='prof' = ICC profile, kind='nclx' = bitstream signalling)
        because the strip path preserves them.

    Box layout reference (ISOBMFF):
      - 4 bytes big-endian uint32 size  (1 = extended, 0 = to EOF)
      - 4 bytes ASCII type
      - [8 bytes uint64 BE size if size==1]
      - body bytes
    ispe body: 4-byte fullbox flags | uint32 BE width | uint32 BE height
    """
    try:
        return _walk_boxes(data, 0, len(data), depth=0)
    except Exception:
        return (0, 0, 0)


def _walk_boxes(
    data: bytes,
    offset: int,
    end: int,
    depth: int,
    width: int = 0,
    height: int = 0,
    meta_bytes: int = 0,
) -> tuple[int, int, int]:
    """Recursive box walker — returns (width, height, meta_bytes)."""
    pos = offset
    while pos + 8 <= end:
        raw_size = struct.unpack_from(">I", data, pos)[0]
        box_type = data[pos + 4 : pos + 8]

        if raw_size == 1:
            # Extended 64-bit size stored in the next 8 bytes
            if pos + 16 > end:
                break
            actual_size = struct.unpack_from(">Q", data, pos + 8)[0]
            header_len = 16
        elif raw_size == 0:
            # Box extends to end of enclosing container
            actual_size = end - pos
            header_len = 8
        else:
            actual_size = raw_size
            header_len = 8

        if actual_size < 8 or pos + actual_size > end:
            break

        body_start = pos + header_len
        body_end = pos + actual_size

        if box_type == b"ispe" and width == 0:
            # ispe body: 4-byte fullbox (version+flags) | uint32 width | uint32 height
            if body_end >= body_start + 12:
                width = struct.unpack_from(">I", data, body_start + 4)[0]
                height = struct.unpack_from(">I", data, body_start + 8)[0]

        elif box_type in _METADATA_BOX_TYPES:
            meta_bytes += actual_size

        elif box_type in _CONTAINER_TYPES and depth < 3:
            inner_start = body_start
            if box_type == b"meta":
                # meta is a FullBox — skip the 4-byte version+flags prefix
                inner_start += 4
            width, height, meta_bytes = _walk_boxes(
                data, inner_start, body_end, depth + 1, width, height, meta_bytes
            )
            # Don't early-return on width != 0 — Exif/xmp siblings of iprp at the
            # meta level still need to be counted toward meta_bytes.

        pos += actual_size

    return width, height, meta_bytes


def _should_skip_avif_optimization(data: bytes, config: OptimizationConfig) -> bool:
    """Return True when box-header analysis predicts no size reduction.

    Conservative thresholds calibrated against the iter3 bench corpus:
    - HIGH (quality < 50):  never skip — re-encode at q=50 reliably shrinks.
    - MEDIUM (50 <= quality < 70): skip if input BPP < 0.5 (AVIF q=70 floor).
    - LOW (quality >= 70, strip-only): skip if strippable metadata < 500 B.
      Strippable = Exif/xml/xmp/mime only; colr boxes are preserved by the
      strip path (kind='prof' ICC kept via icc_profile kwarg, kind='nclx' lives
      in the bitstream) and are therefore NOT counted toward the threshold.

    Skipping saves the full libavif decode cost (~300-2000 ms per megapixel).
    """
    if config.quality < 50:
        # HIGH preset always runs; re-encode wins 100% of the time in bench data.
        return False

    width, height, meta_bytes = _parse_avif_metadata(data)

    if width == 0 or height == 0:
        # Parse failed — fall back to decode path rather than incorrectly skipping.
        return False

    n_pixels = width * height

    if config.quality < 70:
        # MEDIUM preset: re-encode at q=70.  Below ~0.5 bpp the re-encode can't
        # shrink further (q=70 is already near the format floor for these inputs).
        # Bench data: 38/41 MEDIUM cases return "none"; all had bpp < 0.5.
        input_bpp = (len(data) * 8) / n_pixels
        return input_bpp < 0.5

    # LOW preset: strip-only path.  Strip saves roughly metadata_bytes minus
    # the container overhead delta.  Below 500 B there is no headroom.
    # Bench data: 41/41 LOW cases return "none"; all had < 200 B metadata.
    if not config.strip_metadata:
        # strip_metadata=False → optimize() would skip strip entirely and hit
        # skip_reencode=True for LOW → returns "none" anyway; fast-path here too.
        return True
    return meta_bytes < 500


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

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """Pre-flight check: skip libavif decode when box headers predict no gain."""
        if _should_skip_avif_optimization(data, config):
            return self._build_result(data, data, "none")
        return await super().optimize(data, config)

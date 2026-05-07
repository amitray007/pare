import asyncio
import io

from PIL import Image

from optimizers.base import BaseOptimizer
from optimizers.utils import binary_search_quality
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class WebpOptimizer(BaseOptimizer):
    """WebP optimization: Pillow re-encode with preset-aware method.

    Pipeline:
    1. Decode image once
    2. Re-encode with Pillow at target quality (method=4 for HIGH, method=3 for MED/LOW)
    3. If max_reduction set and exceeded, binary search quality (reuses decoded img)

    method=4 is kept for HIGH (quality < 50) to maximise compression ratio.
    method=3 is used for MEDIUM/LOW (quality >= 50) — ~25-30% faster encode at
    the cost of ~1% larger output, acceptable given the SSIMULACRA2 headroom.
    """

    format = ImageFormat.WEBP

    @staticmethod
    def _webp_method(quality: int) -> int:
        """Return the libwebp encode method for a given quality value.

        HIGH preset (quality < 50): method=4 — maximum compression effort.
        MEDIUM/LOW presets (quality >= 50): method=3 — ~25-30% faster encode,
        ~1% larger output, acceptable given ssimu2 headroom on those presets.
        """
        return 4 if quality < 50 else 3

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        # Decode once, share across all paths
        img, is_animated = await asyncio.to_thread(self._decode_image, data)

        enc_method = self._webp_method(config.quality)
        best = await asyncio.to_thread(
            self._encode_webp, img, config.quality, is_animated, enc_method
        )
        method = f"pillow-m{enc_method}"

        # Cap reduction if max_reduction is set (reuses pre-decoded img)
        if config.max_reduction is not None:
            reduction = (1 - len(best) / len(data)) * 100
            if reduction > config.max_reduction:
                capped = await asyncio.to_thread(
                    self._find_capped_quality, img, is_animated, data, config, enc_method
                )
                if capped is not None:
                    best = capped
                    method = f"pillow-m{enc_method}"

        return self._build_result(data, best, method)

    @staticmethod
    def _decode_image(data: bytes) -> tuple[Image.Image, bool]:
        """Decode WebP once. Returns (img, is_animated)."""
        img = Image.open(io.BytesIO(data))
        is_animated = getattr(img, "n_frames", 1) > 1
        return img, is_animated

    def _find_capped_quality(
        self,
        img: Image.Image,
        is_animated: bool,
        data: bytes,
        config: OptimizationConfig,
        enc_method: int,
    ) -> bytes | None:
        """Binary search Pillow quality to cap reduction at max_reduction.

        enc_method is passed through so the search uses the same libwebp method
        as the initial encode — keeping the size relationship coherent.
        """

        def encode_fn(quality: int) -> bytes:
            return self._encode_webp(img, quality, is_animated, enc_method)

        return binary_search_quality(
            encode_fn, len(data), config.max_reduction, lo=config.quality, hi=100
        )

    @staticmethod
    def _encode_webp(img: Image.Image, quality: int, is_animated: bool, method: int = 4) -> bytes:
        """Encode a Pillow Image to WebP bytes.

        Args:
            img: Decoded Pillow image.
            quality: WebP quality value (1-100).
            is_animated: Whether the image has multiple frames.
            method: libwebp encode method (3 = faster, 4 = more compressed).
        """
        if is_animated:
            img.seek(0)  # Reset frame pointer before re-encode

        output = io.BytesIO()

        save_kwargs = {
            "format": "WEBP",
            "quality": quality,
            "method": method,
        }

        if is_animated:
            save_kwargs["save_all"] = True
            save_kwargs["minimize_size"] = True

        img.save(output, **save_kwargs)
        return output.getvalue()

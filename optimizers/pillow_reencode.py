"""Shared base class for Pillow-based re-encoding optimizers (AVIF, HEIC, JXL).

These three formats share the same optimization strategy:
1. Try metadata stripping (lossless re-encode without metadata)
2. Try lossy re-encoding at target quality
3. Pick the smallest result

Subclasses define format-specific constants (quality range, save format,
extra kwargs) and a plugin import hook. All shared logic lives here.
"""

import asyncio
import io
import logging

from PIL import Image

from optimizers.base import BaseOptimizer
from optimizers.utils import clamp_quality
from schemas import OptimizationConfig, OptimizeResult

logger = logging.getLogger("pare.optimizers")


class PillowReencodeOptimizer(BaseOptimizer):
    """Base optimizer for formats that optimize via Pillow strip + re-encode.

    Subclasses MUST set these class attributes:
        format: ImageFormat enum value
        pillow_format: Pillow save format string ("AVIF", "HEIF", "JXL")
        strip_method_name: Method name reported for metadata strip results
        reencode_method_name: Method name reported for re-encode results
        quality_min: Minimum clamped quality (default 30)
        quality_max: Maximum clamped quality (default 90)
        quality_offset: Added to input quality before clamping (default 10)

    Subclasses MAY set:
        extra_save_kwargs: Additional kwargs passed to Pillow save (default {})

    Subclasses MUST override:
        _ensure_plugin(): Import/register the format's Pillow plugin.

    Subclasses MAY override:
        _open_image(data): Custom image loading (e.g. HEIC uses pillow-heif).
    """

    pillow_format: str
    strip_method_name: str
    reencode_method_name: str
    quality_min: int = 30
    quality_max: int = 90
    quality_offset: int = 10
    extra_save_kwargs: dict | None = None

    def _ensure_plugin(self) -> None:
        """Import/register the format's Pillow plugin.

        Called before any Pillow operation. Subclasses MUST override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _ensure_plugin() "
            f"to register the Pillow plugin for {self.pillow_format}"
        )

    def _open_image(self, data: bytes) -> Image.Image:
        """Open image from bytes. Override for formats needing special loading."""
        self._ensure_plugin()
        return Image.open(io.BytesIO(data))

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """Run metadata strip and lossy re-encode concurrently, pick smallest."""
        img = await asyncio.to_thread(self._open_image, data)

        tasks = []
        method_names = []

        if config.strip_metadata:
            tasks.append(asyncio.to_thread(self._strip_metadata_from_img, img.copy(), data))
            method_names.append(self.strip_method_name)

        tasks.append(asyncio.to_thread(self._reencode_from_img, img, config.quality))
        method_names.append(self.reencode_method_name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        for result, method in zip(results, method_names):
            if isinstance(result, BaseException):
                logger.warning(
                    "%s method %s failed: %s: %s",
                    self.pillow_format,
                    method,
                    type(result).__name__,
                    result,
                )
            else:
                candidates.append((result, method))

        if not candidates:
            logger.error(
                "All optimization methods failed for %s, returning original",
                self.pillow_format,
            )
            return self._build_result(data, data, "none")

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)

    def _strip_metadata_from_img(self, img: Image.Image, original_data: bytes) -> bytes:
        """Strip metadata from a pre-decoded Image, preserving ICC profile."""
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": self.pillow_format, "lossless": True}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        return result if len(result) < len(original_data) else original_data

    def _strip_metadata(self, data: bytes) -> bytes:
        """Strip metadata by lossless re-encode, preserving ICC profile.

        Convenience wrapper that decodes from bytes. Prefer _strip_metadata_from_img
        when an Image is already available.
        """
        img = self._open_image(data)
        return self._strip_metadata_from_img(img, data)

    def _reencode_from_img(self, img: Image.Image, quality: int) -> bytes:
        """Re-encode a pre-decoded Image at target quality."""
        icc_profile = img.info.get("icc_profile")

        mapped_quality = clamp_quality(
            quality, offset=self.quality_offset, lo=self.quality_min, hi=self.quality_max
        )

        output = io.BytesIO()
        save_kwargs = {
            "format": self.pillow_format,
            "quality": mapped_quality,
            **(self.extra_save_kwargs or {}),
        }
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        return output.getvalue()

    def _reencode(self, data: bytes, quality: int) -> bytes:
        """Re-encode at target quality with format-specific settings.

        Convenience wrapper that decodes from bytes. Prefer _reencode_from_img
        when an Image is already available.
        """
        img = self._open_image(data)
        return self._reencode_from_img(img, quality)

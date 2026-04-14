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

# Images above this threshold run strip+reencode sequentially
# to avoid memory-expensive img.copy() calls.
PARALLEL_PIXEL_THRESHOLD = 5_000_000  # 5 megapixels


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
        """Run metadata strip and lossy re-encode, pick smallest.

        For images >5MP, runs sequentially to avoid img.copy() memory overhead.
        For small images, runs concurrently (parallel with copy).
        """
        img = await asyncio.to_thread(self._open_image, data)

        pixel_count = img.size[0] * img.size[1]
        use_parallel = pixel_count < PARALLEL_PIXEL_THRESHOLD

        # For lossless presets (quality >= 70) with strip enabled, skip lossy reencode.
        # The strip path (lossless re-encode) is almost always better at high quality.
        skip_reencode = config.quality >= 70 and config.strip_metadata

        # No methods would run — return original immediately
        if skip_reencode and not config.strip_metadata:
            return self._build_result(data, data, "none")

        if use_parallel:
            results, method_names = await self._optimize_parallel(img, data, config, skip_reencode)
        else:
            results, method_names = await self._optimize_sequential(
                img, data, config, skip_reencode
            )

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

    async def _optimize_parallel(
        self,
        img: Image.Image,
        data: bytes,
        config: OptimizationConfig,
        skip_reencode: bool = False,
    ) -> tuple[list, list[str]]:
        """Small image: run strip and reencode concurrently with img.copy()."""
        tasks = []
        method_names = []

        if config.strip_metadata:
            tasks.append(asyncio.to_thread(self._strip_metadata_from_img, img.copy(), data))
            method_names.append(self.strip_method_name)

        if not skip_reencode:
            tasks.append(asyncio.to_thread(self._reencode_from_img, img, config.quality))
            method_names.append(self.reencode_method_name)

        results = list(await asyncio.gather(*tasks, return_exceptions=True))
        return results, method_names

    async def _optimize_sequential(
        self,
        img: Image.Image,
        data: bytes,
        config: OptimizationConfig,
        skip_reencode: bool = False,
    ) -> tuple[list, list[str]]:
        """Large image: run strip then reencode sequentially on shared img."""
        results = []
        method_names = []

        if config.strip_metadata:
            try:
                strip_result = await asyncio.to_thread(self._strip_metadata_from_img, img, data)
                results.append(strip_result)
            except Exception as e:
                results.append(e)
            method_names.append(self.strip_method_name)

        if not skip_reencode:
            try:
                reencode_result = await asyncio.to_thread(
                    self._reencode_from_img, img, config.quality
                )
                results.append(reencode_result)
            except Exception as e:
                results.append(e)
            method_names.append(self.reencode_method_name)

        return results, method_names

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

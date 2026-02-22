import asyncio
import io

from PIL import Image

from config import settings
from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.subprocess_runner import run_tool


class JpegOptimizer(BaseOptimizer):
    """JPEG optimization: Pillow/jpegli (lossy) + jpegtran (lossless).

    Pipeline:
    1. Decode JPEG once via Pillow
    2. Run in parallel:
       a. Pillow JPEG encode at target quality → "jpegli"
       b. jpegtran subprocess (lossless Huffman optimization)
    3. Cap lossy if max_reduction set (binary search in-process)
    4. Pick smallest result
    5. Enforce optimization guarantee (output <= input)

    In Docker, jpegli's libjpeg.so.62 is loaded via ldconfig so Pillow
    uses jpegli automatically. Locally, Pillow uses libjpeg-turbo.

    Set JPEG_ENCODER=cjpeg to fall back to MozJPEG subprocess pipeline.
    """

    format = ImageFormat.JPEG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        if settings.jpeg_encoder == "cjpeg":
            return await self._optimize_cjpeg(data, config)

        # Decode JPEG once
        img, icc_profile, exif_bytes = await asyncio.to_thread(
            self._decode_image, data, config.strip_metadata
        )

        # Run Pillow encode and jpegtran concurrently
        pillow_out, jpegtran_out = await asyncio.gather(
            asyncio.to_thread(
                self._pillow_encode,
                img,
                config.quality,
                config.progressive_jpeg,
                icc_profile,
                exif_bytes,
            ),
            self._run_jpegtran(data, config.progressive_jpeg),
        )

        # Cap lossy if max_reduction is set.
        # Jpegtran (lossless) is never capped — no quality loss.
        if config.max_reduction is not None:
            lossy_red = (1 - len(pillow_out) / len(data)) * 100
            if lossy_red > config.max_reduction:
                capped = await asyncio.to_thread(
                    self._cap_quality, img, len(data), config, icc_profile, exif_bytes
                )
                if capped is not None:
                    pillow_out = capped
                else:
                    pillow_out = data  # Can't fit within cap

        # Pick smallest between (possibly capped) Pillow and jpegtran
        candidates = [(pillow_out, "jpegli"), (jpegtran_out, "jpegtran")]
        best_data, best_method = min(candidates, key=lambda x: len(x[0]))

        return self._build_result(data, best_data, best_method)

    def _decode_image(
        self, data: bytes, strip_metadata: bool
    ) -> tuple[Image.Image, bytes | None, bytes | None]:
        """Decode JPEG and extract metadata for preservation."""
        img = Image.open(io.BytesIO(data))

        # Extract metadata before conversion
        icc_profile = None
        exif_bytes = None
        if not strip_metadata:
            icc_profile = img.info.get("icc_profile")
            exif_bytes = img.info.get("exif")

        # Convert mode if needed
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        return img, icc_profile, exif_bytes

    def _pillow_encode(
        self,
        img: Image.Image,
        quality: int,
        progressive: bool,
        icc_profile: bytes | None,
        exif_bytes: bytes | None,
    ) -> bytes:
        """In-process JPEG encode via Pillow (uses jpegli in Docker)."""
        buf = io.BytesIO()
        save_kwargs: dict = {
            "format": "JPEG",
            "quality": quality,
            "optimize": True,
        }
        if progressive:
            save_kwargs["progressive"] = True
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        img.save(buf, **save_kwargs)
        return buf.getvalue()

    def _cap_quality(
        self,
        img: Image.Image,
        original_size: int,
        config: OptimizationConfig,
        icc_profile: bytes | None,
        exif_bytes: bytes | None,
    ) -> bytes | None:
        """Binary search Pillow quality to cap lossy reduction at max_reduction.

        Returns the encoded output at the lowest quality that stays within
        the cap, or None if even q=100 exceeds the cap.
        """
        target = config.max_reduction

        out_100 = self._pillow_encode(img, 100, config.progressive_jpeg, icc_profile, exif_bytes)
        red_100 = (1 - len(out_100) / original_size) * 100
        if red_100 > target:
            return None  # Even q=100 exceeds cap

        lo, hi = config.quality, 100
        best_out = out_100

        for _ in range(5):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            out_mid = self._pillow_encode(
                img, mid, config.progressive_jpeg, icc_profile, exif_bytes
            )
            red_mid = (1 - len(out_mid) / original_size) * 100
            if red_mid > target:
                lo = mid
            else:
                hi = mid
                best_out = out_mid

        return best_out

    async def _run_jpegtran(self, data: bytes, progressive: bool) -> bytes:
        """Run jpegtran for lossless Huffman table optimization.

        Returns original data if jpegtran is not installed, so the Pillow
        path always wins the "pick smallest" comparison.
        """
        cmd = ["jpegtran", "-optimize", "-copy", "none"]
        if progressive:
            cmd.append("-progressive")
        try:
            stdout, stderr, rc = await run_tool(cmd, data)
            return stdout
        except (FileNotFoundError, OSError):
            return data

    # --- Legacy cjpeg fallback (JPEG_ENCODER=cjpeg) ---

    async def _optimize_cjpeg(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """MozJPEG subprocess pipeline (legacy fallback)."""
        bmp_data = await asyncio.to_thread(self._decode_to_bmp, data, config.strip_metadata)

        mozjpeg_out, jpegtran_out = await asyncio.gather(
            self._run_cjpeg(bmp_data, config.quality, config.progressive_jpeg),
            self._run_jpegtran(data, config.progressive_jpeg),
        )

        if config.max_reduction is not None:
            moz_red = (1 - len(mozjpeg_out) / len(data)) * 100
            if moz_red > config.max_reduction:
                mozjpeg_out = await self._cap_mozjpeg(bmp_data, data, config)

        candidates = [(mozjpeg_out, "mozjpeg"), (jpegtran_out, "jpegtran")]
        best_data, best_method = min(candidates, key=lambda x: len(x[0]))

        return self._build_result(data, best_data, best_method)

    async def _cap_mozjpeg(
        self,
        bmp_data: bytes,
        original: bytes,
        config: OptimizationConfig,
    ) -> bytes:
        """Binary search cjpeg quality to cap lossy reduction at max_reduction."""
        target = config.max_reduction
        orig_size = len(original)

        out_100 = await self._run_cjpeg(bmp_data, 100, config.progressive_jpeg)
        red_100 = (1 - len(out_100) / orig_size) * 100
        if red_100 > target:
            return original

        lo, hi = config.quality, 100
        best_out = out_100

        for _ in range(5):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            out_mid = await self._run_cjpeg(bmp_data, mid, config.progressive_jpeg)
            red_mid = (1 - len(out_mid) / orig_size) * 100
            if red_mid > target:
                lo = mid
            else:
                hi = mid
                best_out = out_mid

        return best_out

    def _decode_to_bmp(self, data: bytes, strip_metadata: bool) -> bytes:
        """Decode JPEG to BMP format for cjpeg input."""
        img = Image.open(io.BytesIO(data))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        output = io.BytesIO()
        img.save(output, format="BMP")
        return output.getvalue()

    async def _run_cjpeg(self, bmp_data: bytes, quality: int, progressive: bool) -> bytes:
        """Run MozJPEG cjpeg on BMP input."""
        cmd = ["cjpeg", "-quality", str(quality)]
        if progressive:
            cmd.append("-progressive")
        stdout, stderr, rc = await run_tool(cmd, bmp_data)
        return stdout

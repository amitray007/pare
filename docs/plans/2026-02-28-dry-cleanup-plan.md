# DRY Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate copy-paste duplication across optimizers and estimation engine by introducing a shared base class, utility functions, and response builders.

**Architecture:** Extract the identical optimize/strip/reencode pattern from AVIF, HEIC, and JXL into `PillowReencodeOptimizer` intermediate base class. Extract binary search quality capping and quality clamping into `optimizers/utils.py`. Consolidate `EstimateResponse` construction into a factory helper. Unify preset definitions to a single source of truth.

**Tech Stack:** Python 3.12, Pillow, pytest, asyncio

**Design doc:** `docs/plans/2026-02-28-dry-cleanup-design.md`

---

### Task 1: Create `optimizers/utils.py` with `clamp_quality`

**Files:**
- Create: `optimizers/utils.py`
- Create: `tests/test_optimizer_utils.py`

**Step 1: Write the failing test**

Create `tests/test_optimizer_utils.py`:

```python
"""Tests for shared optimizer utilities."""

from optimizers.utils import clamp_quality


def test_clamp_quality_default():
    """Default offset=10, lo=30, hi=90."""
    assert clamp_quality(40) == 50
    assert clamp_quality(80) == 90
    assert clamp_quality(15) == 30  # clamped to lo
    assert clamp_quality(95) == 90  # clamped to hi


def test_clamp_quality_custom_range():
    """JXL uses hi=95."""
    assert clamp_quality(85, hi=95) == 95
    assert clamp_quality(90, hi=95) == 95  # 90+10=100, clamped to 95


def test_clamp_quality_custom_offset():
    """Custom offset."""
    assert clamp_quality(50, offset=0) == 50
    assert clamp_quality(50, offset=20) == 70
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_optimizer_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'optimizers.utils'`

**Step 3: Write minimal implementation**

Create `optimizers/utils.py`:

```python
"""Shared utilities for format-specific optimizers."""


def clamp_quality(quality: int, *, offset: int = 10, lo: int = 30, hi: int = 90) -> int:
    """Map Pare quality (1-100, lower=aggressive) to format-specific quality.

    Each format encoder has its own quality scale. This function applies a
    linear offset and clamps to the format's valid range.

    Args:
        quality: Pare quality value (1-100).
        offset: Added to quality before clamping.
        lo: Minimum output quality.
        hi: Maximum output quality.

    Returns:
        Clamped quality value for the format encoder.
    """
    return max(lo, min(hi, quality + offset))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_optimizer_utils.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add optimizers/utils.py tests/test_optimizer_utils.py
git commit -m "feat: add clamp_quality utility to optimizers/utils.py"
```

---

### Task 2: Add `binary_search_quality` to `optimizers/utils.py`

**Files:**
- Modify: `optimizers/utils.py`
- Modify: `tests/test_optimizer_utils.py`

**Step 1: Write the failing test**

Append to `tests/test_optimizer_utils.py`:

```python
from optimizers.utils import binary_search_quality


def test_binary_search_finds_quality_within_cap():
    """Binary search finds quality where reduction stays under target."""
    # Simulate: encode_fn returns smaller output at lower quality
    def encode_fn(quality: int) -> bytes:
        # Lower quality -> smaller output (linear simulation)
        size = int(1000 * (quality / 100))
        return b"x" * max(1, size)

    original_size = 1000
    target_reduction = 30.0  # cap at 30% reduction

    result = binary_search_quality(encode_fn, original_size, target_reduction, lo=40, hi=100)
    assert result is not None
    reduction = (1 - len(result) / original_size) * 100
    assert reduction <= target_reduction + 1.0  # small tolerance


def test_binary_search_returns_none_when_q100_exceeds():
    """Returns None when even q=100 exceeds the cap."""
    def encode_fn(quality: int) -> bytes:
        # Even q=100 produces 50% reduction
        return b"x" * 500

    result = binary_search_quality(encode_fn, 1000, target_reduction=10.0, lo=40, hi=100)
    assert result is None


def test_binary_search_max_iterations():
    """Respects max_iters limit."""
    call_count = 0

    def encode_fn(quality: int) -> bytes:
        nonlocal call_count
        call_count += 1
        size = int(1000 * (quality / 100))
        return b"x" * max(1, size)

    binary_search_quality(encode_fn, 1000, target_reduction=30.0, lo=1, hi=100, max_iters=3)
    # 1 call for q=100 check + up to 3 iterations
    assert call_count <= 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_optimizer_utils.py::test_binary_search_finds_quality_within_cap -v`
Expected: FAIL — `ImportError: cannot import name 'binary_search_quality'`

**Step 3: Write minimal implementation**

Append to `optimizers/utils.py`:

```python
def binary_search_quality(
    encode_fn,
    original_size: int,
    target_reduction: float,
    lo: int,
    hi: int,
    max_iters: int = 5,
) -> bytes | None:
    """Binary search for the lowest quality whose output stays within a reduction cap.

    Used by JPEG and WebP optimizers to enforce max_reduction. The search
    finds the lowest quality (= most compression) that doesn't exceed the
    target reduction percentage.

    Args:
        encode_fn: Callable(quality: int) -> bytes. Format-specific encoder.
        original_size: Size of the original file in bytes.
        target_reduction: Maximum allowed reduction percentage (0-100).
        lo: Lower bound of quality range.
        hi: Upper bound of quality range.
        max_iters: Maximum binary search iterations (default 5).

    Returns:
        Encoded bytes at the capped quality, or None if even q=hi exceeds the cap.
    """
    out_hi = encode_fn(hi)
    red_hi = (1 - len(out_hi) / original_size) * 100
    if red_hi > target_reduction:
        return None  # Even highest quality exceeds cap

    best_out = out_hi

    for _ in range(max_iters):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        out_mid = encode_fn(mid)
        red_mid = (1 - len(out_mid) / original_size) * 100
        if red_mid > target_reduction:
            lo = mid
        else:
            hi = mid
            best_out = out_mid

    return best_out
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_optimizer_utils.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add optimizers/utils.py tests/test_optimizer_utils.py
git commit -m "feat: add binary_search_quality utility to optimizers/utils.py"
```

---

### Task 3: Create `PillowReencodeOptimizer` base class

**Files:**
- Create: `optimizers/pillow_reencode.py`
- Create: `tests/test_pillow_reencode.py`

**Step 1: Write the failing test**

Create `tests/test_pillow_reencode.py`:

```python
"""Tests for PillowReencodeOptimizer shared base class."""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from optimizers.pillow_reencode import PillowReencodeOptimizer
from schemas import OptimizationConfig
from utils.format_detect import ImageFormat


class FakeReencodeOptimizer(PillowReencodeOptimizer):
    """Concrete subclass for testing the base class logic."""

    format = ImageFormat.AVIF
    pillow_format = "PNG"  # Use PNG to avoid needing pillow_avif
    strip_method_name = "test-strip"
    reencode_method_name = "test-reencode"
    quality_min = 30
    quality_max = 90
    quality_offset = 10

    def _ensure_plugin(self):
        pass  # No plugin needed for PNG


def _make_png(size=(100, 100), color=(128, 64, 32)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_optimizer():
    return FakeReencodeOptimizer()


@pytest.mark.asyncio
async def test_optimize_returns_valid_result(fake_optimizer):
    """optimize() returns a valid OptimizeResult."""
    data = _make_png()
    config = OptimizationConfig(quality=60)
    result = await fake_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
async def test_optimize_with_metadata_strip(fake_optimizer):
    """Both strip and reencode run when strip_metadata=True."""
    data = _make_png()
    config = OptimizationConfig(quality=60, strip_metadata=True)
    result = await fake_optimizer.optimize(data, config)
    assert result.success
    assert result.method in ("test-strip", "test-reencode", "none")


@pytest.mark.asyncio
async def test_optimize_without_metadata_strip(fake_optimizer):
    """Only reencode runs when strip_metadata=False."""
    data = _make_png()
    config = OptimizationConfig(quality=60, strip_metadata=False)
    result = await fake_optimizer.optimize(data, config)
    assert result.success
    assert result.method in ("test-reencode", "none")


@pytest.mark.asyncio
async def test_optimize_both_fail_returns_none(fake_optimizer):
    """When both methods fail, returns method='none'."""
    data = _make_png()
    config = OptimizationConfig(quality=60, strip_metadata=True)

    with patch.object(fake_optimizer, "_strip_metadata", side_effect=Exception("fail")):
        with patch.object(fake_optimizer, "_reencode", side_effect=Exception("fail")):
            result = await fake_optimizer.optimize(data, config)
            assert result.method == "none"
            assert result.success


def test_strip_metadata_preserves_icc(fake_optimizer):
    """_strip_metadata preserves ICC profile if present."""
    img = Image.new("RGB", (50, 50), (100, 100, 100))
    # Create a minimal ICC profile in the image info
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    result = fake_optimizer._strip_metadata(data)
    assert isinstance(result, bytes)
    assert len(result) <= len(data)


def test_reencode_uses_clamped_quality(fake_optimizer):
    """_reencode clamps quality using the subclass's range."""
    data = _make_png()
    # quality=15 + offset=10 = 25, clamped to lo=30
    result = fake_optimizer._reencode(data, 15)
    assert isinstance(result, bytes)
    assert len(result) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pillow_reencode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'optimizers.pillow_reencode'`

**Step 3: Write minimal implementation**

Create `optimizers/pillow_reencode.py`:

```python
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

from PIL import Image

from optimizers.base import BaseOptimizer
from optimizers.utils import clamp_quality
from schemas import OptimizationConfig, OptimizeResult


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
    extra_save_kwargs: dict = {}

    def _ensure_plugin(self) -> None:
        """Import/register the format's Pillow plugin.

        Called before any Pillow operation. Override in subclasses.
        """

    def _open_image(self, data: bytes) -> Image.Image:
        """Open image from bytes. Override for formats needing special loading."""
        self._ensure_plugin()
        return Image.open(io.BytesIO(data))

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """Run metadata strip and lossy re-encode concurrently, pick smallest."""
        tasks = []
        method_names = []

        if config.strip_metadata:
            tasks.append(asyncio.to_thread(self._strip_metadata, data))
            method_names.append(self.strip_method_name)

        tasks.append(asyncio.to_thread(self._reencode, data, config.quality))
        method_names.append(self.reencode_method_name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        for result, method in zip(results, method_names):
            if not isinstance(result, Exception):
                candidates.append((result, method))

        if not candidates:
            return self._build_result(data, data, "none")

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)

    def _strip_metadata(self, data: bytes) -> bytes:
        """Strip metadata by lossless re-encode, preserving ICC profile."""
        img = self._open_image(data)
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": self.pillow_format, "lossless": True}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        return result if len(result) < len(data) else data

    def _reencode(self, data: bytes, quality: int) -> bytes:
        """Re-encode at target quality with format-specific settings."""
        img = self._open_image(data)
        icc_profile = img.info.get("icc_profile")

        mapped_quality = clamp_quality(
            quality, offset=self.quality_offset, lo=self.quality_min, hi=self.quality_max
        )

        output = io.BytesIO()
        save_kwargs = {
            "format": self.pillow_format,
            "quality": mapped_quality,
            **self.extra_save_kwargs,
        }
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        return output.getvalue()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pillow_reencode.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add optimizers/pillow_reencode.py tests/test_pillow_reencode.py
git commit -m "feat: add PillowReencodeOptimizer base class for AVIF/HEIC/JXL"
```

---

### Task 4: Migrate AvifOptimizer to PillowReencodeOptimizer

**Files:**
- Modify: `optimizers/avif.py`
- Test: `tests/test_optimizer_avif.py` (existing — must still pass)

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_optimizer_avif.py tests/test_formats.py -k avif -v`
Expected: PASS (or SKIP if pillow_avif not installed)

**Step 2: Rewrite AvifOptimizer**

Replace `optimizers/avif.py` with:

```python
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

    def _strip_metadata(self, data: bytes) -> bytes:
        """AVIF strip uses quality=100 instead of lossless=True."""
        import pillow_avif  # noqa: F401

        import io
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "AVIF", "quality": 100}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        return result if len(result) < len(data) else data
```

Note: AVIF uses `quality=100` for lossless strip instead of `lossless=True` (pillow-avif-plugin convention). This override is needed.

**Step 3: Run tests to verify no regression**

Run: `pytest tests/test_optimizer_avif.py tests/test_formats.py -k avif -v`
Expected: PASS (same results as baseline)

**Step 4: Commit**

```bash
git add optimizers/avif.py
git commit -m "refactor: migrate AvifOptimizer to PillowReencodeOptimizer base"
```

---

### Task 5: Migrate HeicOptimizer to PillowReencodeOptimizer

**Files:**
- Modify: `optimizers/heic.py`
- Test: `tests/test_optimizer_heic.py` (existing — must still pass)

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_optimizer_heic.py tests/test_formats.py -k heic -v`
Expected: PASS (or SKIP if pillow_heif not installed)

**Step 2: Rewrite HeicOptimizer**

Replace `optimizers/heic.py` with:

```python
import io

import pillow_heif
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
        pillow_heif.register_heif_opener()

    def _open_image(self, data: bytes) -> Image.Image:
        """HEIC uses pillow-heif's direct decoder for reliable loading."""
        self._ensure_plugin()
        heif_file = pillow_heif.open_heif(data)
        return heif_file.to_pillow()

    def _strip_metadata(self, data: bytes) -> bytes:
        """HEIC strip uses quality=-1 (lossless) via pillow-heif."""
        self._ensure_plugin()
        heif_file = pillow_heif.open_heif(data)
        img = heif_file.to_pillow()
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "HEIF", "quality": -1}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        return result if len(result) < len(data) else data
```

Note: HEIC has two format-specific differences:
1. `_open_image` uses `pillow_heif.open_heif()` → `.to_pillow()` instead of `Image.open()`
2. `_strip_metadata` uses `quality=-1` for lossless instead of `lossless=True`

**Step 3: Run tests to verify no regression**

Run: `pytest tests/test_optimizer_heic.py tests/test_formats.py -k heic -v`
Expected: PASS (same results as baseline)

**Step 4: Commit**

```bash
git add optimizers/heic.py
git commit -m "refactor: migrate HeicOptimizer to PillowReencodeOptimizer base"
```

---

### Task 6: Migrate JxlOptimizer to PillowReencodeOptimizer

**Files:**
- Modify: `optimizers/jxl.py`
- Test: `tests/test_optimizer_jxl.py` (existing — must still pass)

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_optimizer_jxl.py tests/test_formats.py -k jxl -v`
Expected: PASS (or SKIP if jxlpy not installed)

**Step 2: Rewrite JxlOptimizer**

Replace `optimizers/jxl.py` with:

```python
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
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            import jxlpy  # noqa: F401
```

JXL is the cleanest migration — the only difference from the base class is `quality_max=95` and the dual-import plugin hook. The base class `_strip_metadata` with `lossless=True` and `_reencode` work directly.

**Step 3: Run tests to verify no regression**

Run: `pytest tests/test_optimizer_jxl.py tests/test_formats.py -k jxl -v`
Expected: PASS (same results as baseline)

**Step 4: Commit**

```bash
git add optimizers/jxl.py
git commit -m "refactor: migrate JxlOptimizer to PillowReencodeOptimizer base"
```

---

### Task 7: Wire `binary_search_quality` into JPEG optimizer

**Files:**
- Modify: `optimizers/jpeg.py`
- Test: `tests/test_optimizer_jpeg_mock.py` (existing — must still pass)

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_optimizer_jpeg_mock.py -v`
Expected: PASS

**Step 2: Refactor `_cap_quality` and `_cap_mozjpeg`**

In `optimizers/jpeg.py`:

1. Add import: `from optimizers.utils import binary_search_quality`

2. Replace `_cap_quality` method (lines 117-154) with:

```python
    def _cap_quality(
        self,
        img: Image.Image,
        original_size: int,
        config: OptimizationConfig,
        icc_profile: bytes | None,
        exif_bytes: bytes | None,
    ) -> bytes | None:
        """Binary search Pillow quality to cap lossy reduction at max_reduction."""
        def encode_fn(quality: int) -> bytes:
            return self._pillow_encode(img, quality, config.progressive_jpeg, icc_profile, exif_bytes)

        return binary_search_quality(
            encode_fn, original_size, config.max_reduction, lo=config.quality, hi=100
        )
```

3. Replace `_cap_mozjpeg` method (lines 192-222) with:

```python
    async def _cap_mozjpeg(
        self,
        bmp_data: bytes,
        original: bytes,
        config: OptimizationConfig,
    ) -> bytes:
        """Binary search cjpeg quality to cap lossy reduction at max_reduction."""
        # _cap_mozjpeg is async but binary_search_quality is sync.
        # Wrap the async cjpeg call for sync context.
        import asyncio

        loop = asyncio.get_event_loop()

        async def _async_encode(quality: int) -> bytes:
            return await self._run_cjpeg(bmp_data, quality, config.progressive_jpeg)

        # Run the binary search with sync wrapper
        out_100 = await self._run_cjpeg(bmp_data, 100, config.progressive_jpeg)
        red_100 = (1 - len(out_100) / len(original)) * 100
        if red_100 > config.max_reduction:
            return original

        lo, hi = config.quality, 100
        best_out = out_100

        for _ in range(5):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            out_mid = await self._run_cjpeg(bmp_data, mid, config.progressive_jpeg)
            red_mid = (1 - len(out_mid) / len(original)) * 100
            if red_mid > config.max_reduction:
                lo = mid
            else:
                hi = mid
                best_out = out_mid

        return best_out
```

Note: `_cap_mozjpeg` stays async because it calls `run_tool()` (async subprocess). The sync `binary_search_quality` utility is used only for the sync `_cap_quality` path. This is acceptable — the mozjpeg path is a legacy fallback and keeping its binary search inline avoids async complexity.

**Step 3: Run tests to verify no regression**

Run: `pytest tests/test_optimizer_jpeg_mock.py -v`
Expected: PASS (same results as baseline)

**Step 4: Commit**

```bash
git add optimizers/jpeg.py
git commit -m "refactor: use binary_search_quality in JPEG _cap_quality"
```

---

### Task 8: Wire `binary_search_quality` into WebP optimizer

**Files:**
- Modify: `optimizers/webp.py`
- Test: `tests/test_optimizer_webp_mock.py` (existing — must still pass)

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_optimizer_webp_mock.py -v`
Expected: PASS

**Step 2: Refactor `_find_capped_quality`**

In `optimizers/webp.py`:

1. Add import: `from optimizers.utils import binary_search_quality`

2. Replace `_find_capped_quality` method (lines 50-80) with:

```python
    def _find_capped_quality(self, data: bytes, config: OptimizationConfig) -> bytes | None:
        """Binary search Pillow quality to cap reduction at max_reduction."""
        def encode_fn(quality: int) -> bytes:
            return self._pillow_optimize(data, quality)

        return binary_search_quality(
            encode_fn, len(data), config.max_reduction, lo=config.quality, hi=100
        )
```

**Step 3: Run tests to verify no regression**

Run: `pytest tests/test_optimizer_webp_mock.py -v`
Expected: PASS (same results as baseline)

**Step 4: Commit**

```bash
git add optimizers/webp.py
git commit -m "refactor: use binary_search_quality in WebP _find_capped_quality"
```

---

### Task 9: Wire `clamp_quality` into estimator BPP helpers

**Files:**
- Modify: `estimation/estimator.py`
- Test: `tests/test_sample_estimator.py` (existing — must still pass)

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: PASS

**Step 2: Update estimator imports and BPP helpers**

In `estimation/estimator.py`:

1. Add import at top: `from optimizers.utils import clamp_quality`

2. Move `import math` and `import subprocess` to module-level imports (after `import io`).

3. Replace quality clamping in `_heic_sample_bpp` (line 408):
   - Before: `heic_quality = max(30, min(90, config.quality + 10))`
   - After: `heic_quality = clamp_quality(config.quality)`

4. Replace quality clamping in `_avif_sample_bpp` (line 431):
   - Before: `avif_quality = max(30, min(90, config.quality + 10))`
   - After: `avif_quality = clamp_quality(config.quality)`

5. Replace quality clamping in `_jxl_sample_bpp` (line 457):
   - Before: `jxl_quality = max(30, min(95, config.quality + 10))`
   - After: `jxl_quality = clamp_quality(config.quality, hi=95)`

6. In `_bpp_to_estimate` (line 319), replace the inline `import math` with the module-level import (already moved in step 2).

7. In `_png_sample_bpp` (line 503), replace the inline `import subprocess` with the module-level import (already moved in step 2).

**Step 3: Run tests to verify no regression**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: PASS (same results as baseline)

**Step 4: Commit**

```bash
git add estimation/estimator.py
git commit -m "refactor: use clamp_quality in estimator BPP helpers, move imports to top"
```

---

### Task 10: Extract `_build_estimate` response factory

**Files:**
- Modify: `estimation/estimator.py`
- Test: `tests/test_sample_estimator.py` (existing — must still pass)

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: PASS

**Step 2: Add the factory helper and refactor construction sites**

In `estimation/estimator.py`, add after `_classify_potential`:

```python
def _build_estimate(
    file_size: int,
    fmt: ImageFormat,
    width: int,
    height: int,
    color_type: str | None,
    bit_depth: int | None,
    estimated_size: int,
    reduction: float,
    method: str,
    confidence: str = "high",
) -> EstimateResponse:
    """Build an EstimateResponse with standard field derivations."""
    return EstimateResponse(
        original_size=file_size,
        original_format=fmt.value,
        dimensions={"width": width, "height": height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=estimated_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=method,
        already_optimized=reduction == 0,
        confidence=confidence,
    )
```

Then replace the 6 `EstimateResponse(...)` construction sites:

**Site 1** — `_estimate_exact` (line 163-175):
```python
    return _build_estimate(
        file_size, fmt, width, height, color_type, bit_depth,
        result.optimized_size, reduction, result.method,
    )
```

**Site 2** — `_estimate_by_sample` optimizer-says-none (lines 255-267):
```python
    return _build_estimate(
        file_size, fmt, width, height, color_type, bit_depth,
        file_size, 0.0, "none",
    )
```

**Site 3** — `_estimate_by_sample` extrapolation (lines 280-292):
```python
    return _build_estimate(
        file_size, fmt, width, height, color_type, bit_depth,
        estimated_size, reduction, result.method,
    )
```

**Site 4** — `_bpp_to_estimate` (lines 348-360):
```python
    return _build_estimate(
        file_size, fmt, width, height, color_type, bit_depth,
        estimated_size, reduction, method,
    )
```

**Site 5** — `estimate_from_thumbnail` already-optimized (lines 676-688):
```python
    return _build_estimate(
        original_file_size, fmt, original_width, original_height, color_type, bit_depth,
        original_file_size, 0.0, "none", confidence="medium",
    )
```

**Site 6** — `estimate_from_thumbnail` extrapolation (lines 698-710):
```python
    return _build_estimate(
        original_file_size, fmt, original_width, original_height, color_type, bit_depth,
        estimated_size, reduction, result.method, confidence="medium",
    )
```

**Step 3: Run tests to verify no regression**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: PASS (same results as baseline)

**Step 4: Commit**

```bash
git add estimation/estimator.py
git commit -m "refactor: extract _build_estimate factory, reduce EstimateResponse duplication"
```

---

### Task 11: Unify preset definitions

**Files:**
- Modify: `estimation/presets.py`
- Test: `tests/test_estimate.py` (existing — must still pass)

**Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_estimate.py -v`
Expected: PASS

**Step 2: Rewrite `estimation/presets.py` to use single source of truth**

Replace `estimation/presets.py` with:

```python
"""Preset -> OptimizationConfig mapping for the estimation API.

Delegates to benchmarks.constants as the single source of truth for preset
definitions. This module provides the get_config_for_preset() convenience
function used by the /estimate endpoint.
"""

from benchmarks.constants import PRESETS_BY_NAME


def get_config_for_preset(preset: str) -> "OptimizationConfig":
    """Convert a preset name to an OptimizationConfig.

    Args:
        preset: "high", "medium", or "low" (case-insensitive).

    Returns:
        OptimizationConfig with appropriate quality and flags.

    Raises:
        ValueError: If preset is not recognized.
    """
    key = preset.upper()
    if key not in PRESETS_BY_NAME:
        raise ValueError(f"Invalid preset: '{preset}'. Must be 'high', 'medium', or 'low'.")
    return PRESETS_BY_NAME[key].config
```

**Step 3: Run tests to verify no regression**

Run: `pytest tests/test_estimate.py -v`
Expected: PASS (same results as baseline — the configs are identical)

**Step 4: Commit**

```bash
git add estimation/presets.py
git commit -m "refactor: unify presets to single source of truth in benchmarks.constants"
```

---

### Task 12: Fix inline imports in `benchmarks/corpus.py`

**Files:**
- Modify: `benchmarks/corpus.py`

**Step 1: Move inline imports to module level**

In `benchmarks/corpus.py`, find all inline `import io` and `from PIL import Image` inside functions and move them to the top of the file with other imports.

**Step 2: Run tests to verify no regression**

Run: `pytest tests/ -v --timeout=60`
Expected: PASS

**Step 3: Commit**

```bash
git add benchmarks/corpus.py
git commit -m "refactor: move inline imports to module level in benchmarks/corpus.py"
```

---

### Task 13: Move `_DIRECT_ENCODE_BPP_FNS` to module level

**Files:**
- Modify: `estimation/estimator.py`

**Step 1: Move the dispatch dict from inside `_estimate_by_sample` to module level**

The dict `_DIRECT_ENCODE_BPP_FNS` (currently at line 219 inside `_estimate_by_sample`) is rebuilt on every call but references module-level functions. Move it to module level, after all the BPP helper function definitions (after `_tiff_sample_bpp`).

**Step 2: Run tests to verify no regression**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add estimation/estimator.py
git commit -m "refactor: move _DIRECT_ENCODE_BPP_FNS to module level"
```

---

### Task 14: Full test suite + lint verification

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `pytest tests/ -v --timeout=120`
Expected: All tests PASS

**Step 2: Run linter**

Run: `python -m ruff check . && python -m black --check .`
Expected: No errors

**Step 3: Fix any issues**

If lint errors appear, fix them (likely formatting from the refactors).

**Step 4: Final commit if fixes needed**

```bash
git add -u
git commit -m "fix: lint and formatting cleanup from DRY refactor"
```

---

### Task 15: Update CLAUDE.md files

**Files:**
- Modify: `optimizers/CLAUDE.md`
- Modify: `estimation/CLAUDE.md`

**Step 1: Update optimizers/CLAUDE.md**

Add documentation for:
- `PillowReencodeOptimizer` pattern and how to use it for new formats
- `optimizers/utils.py` utilities (`clamp_quality`, `binary_search_quality`)
- Updated "How to Add a New Optimizer" section mentioning the base class option

**Step 2: Update estimation/CLAUDE.md**

Update the "Direct-Encode BPP Helpers" section to note that quality clamping now uses `clamp_quality` from `optimizers/utils.py`.

**Step 3: Commit**

```bash
git add optimizers/CLAUDE.md estimation/CLAUDE.md
git commit -m "docs: update CLAUDE.md files for DRY refactor"
```

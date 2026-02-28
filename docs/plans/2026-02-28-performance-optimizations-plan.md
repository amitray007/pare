# Performance Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate redundant image decoding in optimizers, reduce BMP memory waste, and move per-request plugin registration to module level.

**Architecture:** Three independent phases — (1) decode-once + copy for AVIF/HEIC/JXL base class and WebP binary search, (2) BMP early-termination palette detection, (3) module-level imports for Pillow plugins and oxipng. Each phase is independently testable and committable.

**Tech Stack:** Python 3.12, Pillow, pillow-heif, pillow-avif-plugin, jxlpy, oxipng, pytest + pytest-asyncio

---

### Task 1: Add decode-once tests for PillowReencodeOptimizer

**Files:**
- Modify: `tests/test_pillow_reencode.py`

**Step 1: Write two new tests**

Add these tests to `tests/test_pillow_reencode.py` after the existing tests:

```python
@pytest.mark.asyncio
async def test_optimize_decodes_image_once(fake_optimizer):
    """optimize() should call _open_image exactly once, not once per method."""
    data = _make_png()
    config = OptimizationConfig(quality=60, strip_metadata=True)

    with patch.object(fake_optimizer, "_open_image", wraps=fake_optimizer._open_image) as mock_open:
        result = await fake_optimizer.optimize(data, config)
        assert result.success
        # Should decode once in optimize(), not once per _strip/_reencode
        assert mock_open.call_count == 1


@pytest.mark.asyncio
async def test_optimize_decodes_once_without_strip(fake_optimizer):
    """optimize() with strip_metadata=False still decodes only once."""
    data = _make_png()
    config = OptimizationConfig(quality=60, strip_metadata=False)

    with patch.object(fake_optimizer, "_open_image", wraps=fake_optimizer._open_image) as mock_open:
        result = await fake_optimizer.optimize(data, config)
        assert result.success
        assert mock_open.call_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pillow_reencode.py::test_optimize_decodes_image_once tests/test_pillow_reencode.py::test_optimize_decodes_once_without_strip -v`

Expected: FAIL — `_open_image` is currently called 2 times (once in `_strip_metadata`, once in `_reencode`) when `strip_metadata=True`, and 1 time when `strip_metadata=False` (but from inside `_reencode`, not from `optimize`).

Note: The second test (`without_strip`) may pass by accident since `_reencode` calls `_open_image` once. That's fine — the first test is the critical one.

**Step 3: Commit the failing tests**

```bash
git add tests/test_pillow_reencode.py
git commit -m "test: add decode-once assertions for PillowReencodeOptimizer"
```

---

### Task 2: Implement decode-once in PillowReencodeOptimizer

**Files:**
- Modify: `optimizers/pillow_reencode.py:63-122`

**Step 1: Refactor optimize() to decode once**

Replace the `optimize`, `_strip_metadata`, and `_reencode` methods in `optimizers/pillow_reencode.py`. The new structure:

1. `optimize()` decodes once via `_open_image()`, passes `img.copy()` to strip and `img` to reencode
2. `_strip_metadata_from_img(img, original_data)` — new method, works from pre-decoded Image
3. `_reencode_from_img(img, quality)` — new method, works from pre-decoded Image
4. Keep old `_strip_metadata(data)` and `_reencode(data, quality)` as thin wrappers (they're called by subclass overrides and tests)

Replace the `optimize` method (lines 63-86) with:

```python
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
            if not isinstance(result, Exception):
                candidates.append((result, method))

        if not candidates:
            return self._build_result(data, data, "none")

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)
```

Replace `_strip_metadata` (lines 88-101) with two methods:

```python
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
```

Replace `_reencode` (lines 103-122) with two methods:

```python
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
            **self.extra_save_kwargs,
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
```

**Step 2: Run the decode-once tests**

Run: `pytest tests/test_pillow_reencode.py -v`

Expected: All 8 tests PASS (6 existing + 2 new decode-once tests).

**Step 3: Run the full test suite to check for regressions**

Run: `pytest tests/ -v`

Expected: All 399+ tests PASS.

**Step 4: Commit**

```bash
git add optimizers/pillow_reencode.py
git commit -m "perf: decode image once in PillowReencodeOptimizer"
```

---

### Task 3: Update AVIF _strip_metadata to use _from_img pattern

**Files:**
- Modify: `optimizers/avif.py:32-47`

**Step 1: Run existing AVIF tests to establish baseline**

Run: `pytest tests/test_optimizer_avif.py tests/test_formats.py::test_format_avif -v`

Expected: All PASS (or skip if pillow_avif not installed).

**Step 2: Refactor AVIF's _strip_metadata override to _strip_metadata_from_img**

Replace the `_strip_metadata` method in `optimizers/avif.py` with:

```python
    def _strip_metadata_from_img(self, img: Image.Image, original_data: bytes) -> bytes:
        """AVIF strip uses quality=100 instead of lossless=True."""
        import pillow_avif  # noqa: F401

        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "AVIF", "quality": 100}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        return result if len(result) < len(original_data) else original_data
```

Also add `from PIL import Image` to the imports if not already there (it is already imported).

**Step 3: Run AVIF tests**

Run: `pytest tests/test_optimizer_avif.py tests/test_formats.py -v`

Expected: All PASS.

**Step 4: Commit**

```bash
git add optimizers/avif.py
git commit -m "perf: AVIF uses decode-once via _strip_metadata_from_img"
```

---

### Task 4: Update HEIC _strip_metadata to use _from_img pattern

**Files:**
- Modify: `optimizers/heic.py:38-53`

**Step 1: Run existing HEIC tests to establish baseline**

Run: `pytest tests/test_optimizer_heic.py tests/test_formats.py -v`

Expected: All PASS (or skip if pillow-heif not installed).

**Step 2: Refactor HEIC's _strip_metadata override to _strip_metadata_from_img**

Replace the `_strip_metadata` method in `optimizers/heic.py` with:

```python
    def _strip_metadata_from_img(self, img: Image.Image, original_data: bytes) -> bytes:
        """HEIC strip uses quality=-1 (lossless) via pillow-heif."""
        self._ensure_plugin()
        icc_profile = img.info.get("icc_profile")

        output = io.BytesIO()
        save_kwargs = {"format": "HEIF", "quality": -1}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        result = output.getvalue()

        return result if len(result) < len(original_data) else original_data
```

Add `from PIL import Image` to the imports if not already there (it is already imported).

**Step 3: Run HEIC tests**

Run: `pytest tests/test_optimizer_heic.py tests/test_formats.py -v`

Expected: All PASS.

**Step 4: Commit**

```bash
git add optimizers/heic.py
git commit -m "perf: HEIC uses decode-once via _strip_metadata_from_img"
```

---

### Task 5: Add WebP decode-once test for binary search

**Files:**
- Modify: `tests/test_optimizer_webp_mock.py`

**Step 1: Write a test that asserts single decode in binary search**

Add this test to `tests/test_optimizer_webp_mock.py`:

```python
def test_find_capped_quality_decodes_once(webp_optimizer):
    """_find_capped_quality should decode the image once, not per iteration."""
    data = _make_webp(quality=95, size=(200, 200))
    config = OptimizationConfig(quality=60, max_reduction=5.0)

    with patch("optimizers.webp.Image.open", wraps=Image.open) as mock_open:
        webp_optimizer._find_capped_quality(data, config)
        # Should decode once, not once per binary search iteration
        assert mock_open.call_count == 1
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/test_optimizer_webp_mock.py::test_find_capped_quality_decodes_once -v`

Expected: FAIL — `Image.open` is currently called 4-7 times (once per binary search iteration via `_pillow_optimize`).

**Step 3: Commit**

```bash
git add tests/test_optimizer_webp_mock.py
git commit -m "test: add decode-once assertion for WebP binary search"
```

---

### Task 6: Implement WebP decode-once in binary search

**Files:**
- Modify: `optimizers/webp.py:51-83`

**Step 1: Refactor _find_capped_quality and _pillow_optimize**

In `optimizers/webp.py`, replace `_find_capped_quality` (lines 51-59) with:

```python
    def _find_capped_quality(self, data: bytes, config: OptimizationConfig) -> bytes | None:
        """Binary search Pillow quality to cap reduction at max_reduction."""
        img = Image.open(io.BytesIO(data))
        is_animated = getattr(img, "n_frames", 1) > 1

        def encode_fn(quality: int) -> bytes:
            return self._encode_webp(img, quality, is_animated)

        return binary_search_quality(
            encode_fn, len(data), config.max_reduction, lo=config.quality, hi=100
        )
```

Extract a shared encode helper and update `_pillow_optimize` to use it. Replace `_pillow_optimize` (lines 61-83) with:

```python
    def _pillow_optimize(self, data: bytes, quality: int) -> bytes:
        """In-process WebP optimization via Pillow.

        Handles both static and animated WebP.
        For animated: preserves all frames via save_all=True.
        """
        img = Image.open(io.BytesIO(data))
        is_animated = getattr(img, "n_frames", 1) > 1
        return self._encode_webp(img, quality, is_animated)

    @staticmethod
    def _encode_webp(img: Image.Image, quality: int, is_animated: bool) -> bytes:
        """Encode a Pillow Image to WebP bytes."""
        output = io.BytesIO()

        save_kwargs = {
            "format": "WEBP",
            "quality": quality,
            "method": 4,
        }

        if is_animated:
            save_kwargs["save_all"] = True
            save_kwargs["minimize_size"] = True

        img.save(output, **save_kwargs)
        return output.getvalue()
```

**Step 2: Run the decode-once test**

Run: `pytest tests/test_optimizer_webp_mock.py::test_find_capped_quality_decodes_once -v`

Expected: PASS — `Image.open` called exactly once.

**Step 3: Run all WebP tests**

Run: `pytest tests/test_optimizer_webp_mock.py tests/test_formats.py::test_format_webp -v`

Expected: All PASS.

**Step 4: Run full test suite**

Run: `pytest tests/ -v`

Expected: All 399+ tests PASS.

**Step 5: Commit**

```bash
git add optimizers/webp.py
git commit -m "perf: decode image once in WebP binary search"
```

---

### Task 7: Add BMP early-termination test

**Files:**
- Modify: `tests/test_formats.py` (add a targeted BMP test)

**Step 1: Write a test for early termination behavior**

Add to `tests/test_formats.py`:

```python
@pytest.mark.asyncio
async def test_bmp_lossless_palette_photographic():
    """Photographic BMP (>256 colors) returns None quickly without full scan."""
    from optimizers.bmp import BmpOptimizer

    # Create an image with >256 unique colors (gradient)
    img = Image.new("RGB", (100, 100))
    for x in range(100):
        for y in range(100):
            img.putpixel((x, y), (x * 2, y * 2, (x + y) % 256))

    result = BmpOptimizer._try_lossless_palette(img)
    assert result is None  # Too many colors
```

Add `from PIL import Image` to the test file imports if not already there.

**Step 2: Run the test**

Run: `pytest tests/test_formats.py::test_bmp_lossless_palette_photographic -v`

Expected: PASS (the current implementation also returns None for >256 colors, just inefficiently).

**Step 3: Write a test that verifies the index map is correct**

```python
@pytest.mark.asyncio
async def test_bmp_lossless_palette_few_colors():
    """BMP with <=256 colors produces valid palette image."""
    from optimizers.bmp import BmpOptimizer

    # Create an image with exactly 4 colors
    img = Image.new("RGB", (10, 10), (255, 0, 0))
    for x in range(5):
        for y in range(5):
            img.putpixel((x, y), (0, 255, 0))
    img.putpixel((9, 9), (0, 0, 255))
    img.putpixel((9, 0), (255, 255, 0))

    result = BmpOptimizer._try_lossless_palette(img)
    assert result is not None
    palette_img, bmp_bytes, method = result
    assert method == "bmp-palette-lossless"
    assert palette_img.mode == "P"
    assert len(bmp_bytes) > 0
```

**Step 4: Run both tests**

Run: `pytest tests/test_formats.py::test_bmp_lossless_palette_photographic tests/test_formats.py::test_bmp_lossless_palette_few_colors -v`

Expected: Both PASS.

**Step 5: Commit**

```bash
git add tests/test_formats.py
git commit -m "test: add BMP palette detection unit tests"
```

---

### Task 8: Implement BMP early-termination palette detection

**Files:**
- Modify: `optimizers/bmp.py:84-121`

**Step 1: Replace _try_lossless_palette with early-termination version**

Replace the `_try_lossless_palette` method (lines 84-121) in `optimizers/bmp.py` with:

```python
    @staticmethod
    def _try_lossless_palette(img: Image.Image) -> tuple[Image.Image, bytes, str] | None:
        """Try lossless conversion to 8-bit palette BMP.

        If the image has <= 256 unique colors, builds an exact palette
        (no quantization, no color loss) and returns (palette_img, bmp_bytes, method).
        Returns None if the image has too many colors.

        Uses single-pass early termination: bails out as soon as color 257 is
        found, and builds the color-to-index map simultaneously.
        """
        unique = {}
        pixel_indices = bytearray()

        for pixel in img.getdata():
            idx = unique.get(pixel)
            if idx is None:
                idx = len(unique)
                if idx >= 256:
                    return None  # Early exit — no wasted allocation
                unique[pixel] = idx
            pixel_indices.append(idx)

        w, h = img.size

        # Build palette image with exact color mapping
        palette_img = Image.new("P", (w, h))
        palette_img.putdata(list(pixel_indices))

        # Build RGB palette (Pillow expects flat R,G,B list of 768 entries)
        flat_palette = [0] * 768
        for color, i in unique.items():
            if isinstance(color, int):
                # Grayscale
                flat_palette[i * 3] = color
                flat_palette[i * 3 + 1] = color
                flat_palette[i * 3 + 2] = color
            else:
                flat_palette[i * 3] = color[0]
                flat_palette[i * 3 + 1] = color[1]
                flat_palette[i * 3 + 2] = color[2]
        palette_img.putpalette(flat_palette)

        buf = io.BytesIO()
        palette_img.save(buf, format="BMP")
        return palette_img, buf.getvalue(), "bmp-palette-lossless"
```

Key differences from the old version:
- `unique` dict maps `{color: index}` instead of building `list(img.getdata())` + `set()` + separate `color_to_idx`
- `bytearray` for pixel indices instead of rebuilding from `pixels` list
- Early `return None` at color 257 instead of scanning all pixels first
- Iterates `unique.items()` (not `enumerate(unique_colors)`) for palette — stable because dict insertion order is preserved in Python 3.7+

**Step 2: Run BMP tests**

Run: `pytest tests/test_formats.py -k bmp -v`

Expected: All BMP tests PASS.

**Step 3: Run full test suite**

Run: `pytest tests/ -v`

Expected: All 399+ tests PASS.

**Step 4: Commit**

```bash
git add optimizers/bmp.py
git commit -m "perf: BMP early-termination palette detection with single pass"
```

---

### Task 9: Move plugin imports to module level

**Files:**
- Modify: `optimizers/avif.py:29-34`
- Modify: `optimizers/heic.py:29-30`
- Modify: `optimizers/jxl.py:24-28`
- Modify: `optimizers/png.py:139`

**Step 1: Run full test suite to establish baseline**

Run: `pytest tests/ -v`

Expected: All PASS.

**Step 2: Move AVIF plugin import to module level**

In `optimizers/avif.py`, add at the top of the file (after the existing imports):

```python
import pillow_avif  # noqa: F401 — registers AVIF codec with Pillow
```

Update `_ensure_plugin` to be a no-op:

```python
    def _ensure_plugin(self):
        pass  # Plugin registered at module import time
```

Remove the `import pillow_avif` line from inside `_strip_metadata_from_img` (if it was added in Task 3 — it's no longer needed since the module-level import handles it).

**Step 3: Move HEIC plugin registration to module level**

In `optimizers/heic.py`, the module-level import `import pillow_heif` already exists. Add the registration call right after:

```python
import pillow_heif
pillow_heif.register_heif_opener()  # Register once at import time
```

Update `_ensure_plugin` to be a no-op:

```python
    def _ensure_plugin(self):
        pass  # Plugin registered at module import time
```

**Step 4: Move JXL plugin import to module level**

In `optimizers/jxl.py`, replace the import section with:

```python
try:
    import pillow_jxl  # noqa: F401 — registers JXL codec with Pillow
except ImportError:
    import jxlpy  # noqa: F401

from optimizers.pillow_reencode import PillowReencodeOptimizer
from utils.format_detect import ImageFormat
```

Update `_ensure_plugin` to be a no-op:

```python
    def _ensure_plugin(self):
        pass  # Plugin registered at module import time
```

**Step 5: Move oxipng import to module level in PNG optimizer**

In `optimizers/png.py`, add at the top with the other imports:

```python
import oxipng
```

Remove the `import oxipng` line from inside `_run_oxipng` (line 139).

The method becomes:

```python
    def _run_oxipng(self, data: bytes, level: int = 2) -> bytes:
        """Run oxipng in-process via pyoxipng library (no subprocess)."""
        return oxipng.optimize_from_memory(data, level=level)
```

**Step 6: Run full test suite**

Run: `pytest tests/ -v`

Expected: All 399+ tests PASS.

**Step 7: Lint check**

Run: `python -m ruff check optimizers/avif.py optimizers/heic.py optimizers/jxl.py optimizers/png.py`

Expected: No errors.

**Step 8: Commit**

```bash
git add optimizers/avif.py optimizers/heic.py optimizers/jxl.py optimizers/png.py
git commit -m "perf: move plugin registration and oxipng import to module level"
```

---

### Task 10: Update CLAUDE.md documentation

**Files:**
- Modify: `optimizers/CLAUDE.md`

**Step 1: Update the PillowReencodeOptimizer section**

In `optimizers/CLAUDE.md`, update the PillowReencodeOptimizer section to mention the decode-once pattern:

Add after "**Optional overrides**: `_open_image(data)` (HEIC uses pillow-heif), `_strip_metadata(data)` (AVIF uses quality=100, HEIC uses quality=-1)":

```
- **Decode-once pattern**: `optimize()` decodes via `_open_image()` once, passes `img.copy()` to strip and `img` to reencode. Override `_strip_metadata_from_img(img, data)` instead of `_strip_metadata(data)` for format-specific strip behavior.
- **Plugin imports**: Moved to module level — `_ensure_plugin()` is a no-op in all current subclasses.
```

**Step 2: Commit**

```bash
git add optimizers/CLAUDE.md
git commit -m "docs: update CLAUDE.md with decode-once pattern and module-level imports"
```

---

### Task 11: Final verification

**Step 1: Run the full test suite**

Run: `pytest tests/ -v`

Expected: All 399+ tests PASS with 0 failures.

**Step 2: Run lint and format checks**

Run: `python -m ruff check . && python -m black --check .`

Expected: No errors.

**Step 3: Review the diff**

Run: `git log --oneline HEAD~10..HEAD` to see all commits from this plan.

Expected: ~8-9 commits covering tests, implementation, and docs for all 3 phases.

"""Tests for WebP optimizer — binary search cap, max_reduction, and method routing."""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from optimizers.webp import WebpOptimizer
from schemas import OptimizationConfig


@pytest.fixture
def webp_optimizer():
    return WebpOptimizer()


def _make_webp(quality=95, size=(100, 100)):
    img = Image.new("RGB", size, (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return buf.getvalue()


def test_find_capped_quality_uses_predecoded_image(webp_optimizer):
    """_find_capped_quality uses a pre-decoded image, no Image.open calls."""
    data = _make_webp(quality=95, size=(200, 200))
    img, is_animated = webp_optimizer._decode_image(data)
    config = OptimizationConfig(quality=60, max_reduction=5.0)

    with patch("optimizers.webp.Image.open") as mock_open:
        webp_optimizer._find_capped_quality(img, is_animated, data, config, enc_method=3)
        # Should NOT call Image.open — uses pre-decoded img
        mock_open.assert_not_called()


@pytest.mark.asyncio
async def test_webp_find_capped_quality_binary_search(webp_optimizer):
    """Binary search finds quality within max_reduction cap."""
    data = _make_webp(quality=95, size=(200, 200))
    img, is_animated = webp_optimizer._decode_image(data)
    config = OptimizationConfig(quality=60, max_reduction=5.0)

    result = webp_optimizer._find_capped_quality(img, is_animated, data, config, enc_method=3)
    # Should return some bytes (binary search found a quality)
    if result is not None:
        reduction = (1 - len(result) / len(data)) * 100
        assert reduction <= 5.0 + 1.0  # small tolerance


@pytest.mark.asyncio
async def test_webp_find_capped_quality_q100_exceeds(webp_optimizer):
    """q=100 still exceeds cap -> returns None."""
    img = Image.new("RGB", (500, 500), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=100)
    data = buf.getvalue()

    decoded_img, is_animated = webp_optimizer._decode_image(data)
    config = OptimizationConfig(quality=1, max_reduction=0.001)

    with patch.object(webp_optimizer, "_encode_webp") as mock_enc:
        # q=100 produces output that's 10% of input -> 90% reduction > 0.001%
        mock_enc.return_value = b"x" * int(len(data) * 0.1)
        result = webp_optimizer._find_capped_quality(
            decoded_img, is_animated, data, config, enc_method=4
        )
    assert result is None


@pytest.mark.asyncio
async def test_webp_max_reduction_triggers_find_capped(webp_optimizer):
    """Optimize path: max_reduction triggers _find_capped_quality."""
    data = _make_webp(quality=95, size=(200, 200))

    with patch.object(webp_optimizer, "_encode_webp") as mock_enc:
        # Pillow returns very small (triggers cap)
        mock_enc.return_value = b"tiny"
        with patch.object(
            webp_optimizer, "_find_capped_quality", return_value=b"capped"
        ) as mock_cap:
            result = await webp_optimizer.optimize(
                data, OptimizationConfig(quality=60, max_reduction=5.0)
            )
    mock_cap.assert_called_once()
    assert result.method == "pillow-m3"


@pytest.mark.asyncio
async def test_webp_optimizer_max_reduction_binary_search():
    """Cover WebP _find_capped_quality binary search break/lo adjustment."""
    opt = WebpOptimizer()

    img = Image.new("RGB", (128, 128), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=100)
    data = buf.getvalue()

    config = OptimizationConfig(quality=20, max_reduction=5.0)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


# ---------------------------------------------------------------------------
# method routing tests
# ---------------------------------------------------------------------------


class TestWebpMethodRouting:
    """Verify _webp_method returns the correct libwebp method for each quality."""

    def test_high_preset_q40_uses_method4(self):
        """quality=40 (HIGH) → method=4 (max compression effort)."""
        assert WebpOptimizer._webp_method(40) == 4

    def test_high_preset_boundary_q49_uses_method4(self):
        """quality=49 (still HIGH) → method=4."""
        assert WebpOptimizer._webp_method(49) == 4

    def test_medium_preset_q60_uses_method3(self):
        """quality=60 (MEDIUM) → method=3 (faster encode)."""
        assert WebpOptimizer._webp_method(60) == 3

    def test_low_preset_q80_uses_method3(self):
        """quality=80 (LOW) → method=3."""
        assert WebpOptimizer._webp_method(80) == 3

    def test_boundary_q50_uses_method3(self):
        """quality=50 is the threshold — >= 50 → method=3."""
        assert WebpOptimizer._webp_method(50) == 3


@pytest.mark.asyncio
async def test_high_preset_encode_called_with_method4(webp_optimizer):
    """optimize() with quality=40 (HIGH) passes method=4 to _encode_webp."""
    data = _make_webp(quality=95, size=(100, 100))

    with patch.object(
        webp_optimizer, "_encode_webp", wraps=webp_optimizer._encode_webp
    ) as mock_enc:
        await webp_optimizer.optimize(data, OptimizationConfig(quality=40))

    # Every _encode_webp call must have method=4
    assert mock_enc.call_count >= 1
    for c in mock_enc.call_args_list:
        assert (
            c.args[3] == 4 or c.kwargs.get("method") == 4
        ), f"Expected method=4 for HIGH preset, got call args: {c}"


@pytest.mark.asyncio
async def test_medium_preset_encode_called_with_method3(webp_optimizer):
    """optimize() with quality=60 (MEDIUM) passes method=3 to _encode_webp."""
    data = _make_webp(quality=95, size=(100, 100))

    with patch.object(
        webp_optimizer, "_encode_webp", wraps=webp_optimizer._encode_webp
    ) as mock_enc:
        await webp_optimizer.optimize(data, OptimizationConfig(quality=60))

    assert mock_enc.call_count >= 1
    for c in mock_enc.call_args_list:
        assert (
            c.args[3] == 3 or c.kwargs.get("method") == 3
        ), f"Expected method=3 for MEDIUM preset, got call args: {c}"


@pytest.mark.asyncio
async def test_low_preset_encode_called_with_method3(webp_optimizer):
    """optimize() with quality=80 (LOW) passes method=3 to _encode_webp."""
    data = _make_webp(quality=95, size=(100, 100))

    with patch.object(
        webp_optimizer, "_encode_webp", wraps=webp_optimizer._encode_webp
    ) as mock_enc:
        await webp_optimizer.optimize(data, OptimizationConfig(quality=80))

    assert mock_enc.call_count >= 1
    for c in mock_enc.call_args_list:
        assert (
            c.args[3] == 3 or c.kwargs.get("method") == 3
        ), f"Expected method=3 for LOW preset, got call args: {c}"


@pytest.mark.asyncio
async def test_max_reduction_path_uses_same_method_as_initial_encode(webp_optimizer):
    """Binary search in _find_capped_quality uses the same method as the initial encode.

    Uses a random-noise image so the encode at q=60 always exceeds max_reduction=5%,
    ensuring the binary-search path is exercised.
    """
    import random

    rng = random.Random(42)
    noise_img = Image.new("RGB", (200, 200))
    noise_img.putdata(
        [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(200 * 200)]
    )
    buf = io.BytesIO()
    noise_img.save(buf, format="WEBP", quality=100)
    data = buf.getvalue()

    config = OptimizationConfig(quality=60, max_reduction=5.0)  # will be exceeded for noise img

    observed_methods: list[int] = []

    original_encode = webp_optimizer._encode_webp

    def capturing_encode(img, quality, is_animated, method=4):
        observed_methods.append(method)
        return original_encode(img, quality, is_animated, method)

    with patch.object(webp_optimizer, "_encode_webp", side_effect=capturing_encode):
        await webp_optimizer.optimize(data, config)

    # All calls (initial + binary search) must have used method=3 (quality=60 >= 50)
    assert len(observed_methods) >= 2, "Expected initial encode + at least one search call"
    assert all(
        m == 3 for m in observed_methods
    ), f"All encode calls should use method=3 for quality=60, got: {observed_methods}"


@pytest.mark.asyncio
async def test_result_method_string_reflects_encode_method(webp_optimizer):
    """OptimizeResult.method includes the libwebp method number for traceability."""
    data = _make_webp(quality=95, size=(100, 100))

    result_high = await webp_optimizer.optimize(data, OptimizationConfig(quality=40))
    result_med = await webp_optimizer.optimize(data, OptimizationConfig(quality=60))

    assert result_high.method in (
        "pillow-m4",
        "none",
    ), f"HIGH preset should report 'pillow-m4', got: {result_high.method}"
    assert result_med.method in (
        "pillow-m3",
        "none",
    ), f"MEDIUM preset should report 'pillow-m3', got: {result_med.method}"

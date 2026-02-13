"""Tests targeting specific coverage gaps across the codebase.

Each test function is named after the module and uncovered line(s) it targets.
"""

import io
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from schemas import OptimizationConfig
from utils.format_detect import ImageFormat


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _jpeg_from_img(img: Image.Image, quality: int = 90) -> bytes:
    """Helper: save a Pillow image as JPEG bytes."""
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L", "CMYK"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_header_info(fmt: ImageFormat, **kwargs):
    """Helper: create a HeaderInfo with defaults."""
    from estimation.header_analysis import HeaderInfo

    info = HeaderInfo(format=fmt)
    for key, val in kwargs.items():
        setattr(info, key, val)
    return info


# ---------------------------------------------------------------------------
# main.py – lifespan startup/shutdown (lines 18-39)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_startup_and_shutdown():
    """Cover main.lifespan: startup tool check + shutdown redis close."""
    from main import lifespan, app

    with patch("routers.health.check_tools", return_value={"pngquant": True, "jpegtran": False}):
        with patch("main.setup_logging"):
            with patch("main.get_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                async with lifespan(app):
                    pass

                # Verify warning logged for missing tool
                mock_logger.warning.assert_called_once()
                assert "Missing tools" in str(mock_logger.warning.call_args)


@pytest.mark.asyncio
async def test_lifespan_no_missing_tools():
    """Cover the branch where all tools are available (no warning logged)."""
    from main import lifespan, app

    all_tools = {
        "pngquant": True, "jpegtran": True, "gifsicle": True,
        "cwebp": True, "oxipng": True, "pillow_heif": True,
        "scour": True, "pillow": True, "jxl_plugin": True,
    }
    with patch("routers.health.check_tools", return_value=all_tools):
        with patch("main.setup_logging"):
            with patch("main.get_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                async with lifespan(app):
                    pass
                mock_logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_shutdown_closes_redis():
    """Cover the shutdown path that closes Redis."""
    from main import lifespan, app

    mock_redis = AsyncMock()
    with patch("routers.health.check_tools", return_value={}):
        with patch("main.setup_logging"):
            with patch("main.get_logger") as mock_get_logger:
                mock_get_logger.return_value = MagicMock()
                with patch("security.rate_limiter._redis", mock_redis):
                    async with lifespan(app):
                        pass
                    mock_redis.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# middleware.py – _get_client_ip with X-Forwarded-For (line 60)
# ---------------------------------------------------------------------------


def test_get_client_ip_forwarded_for():
    """Cover middleware._get_client_ip parsing X-Forwarded-For header."""
    from middleware import _get_client_ip

    mock_request = MagicMock()
    mock_request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8, 9.10.11.12"}
    mock_request.client.host = "127.0.0.1"

    ip = _get_client_ip(mock_request)
    assert ip == "1.2.3.4"


def test_get_client_ip_single_forwarded():
    """Cover single IP in X-Forwarded-For."""
    from middleware import _get_client_ip

    mock_request = MagicMock()
    mock_request.headers = {"X-Forwarded-For": " 10.0.0.1 "}
    mock_request.client.host = "127.0.0.1"

    ip = _get_client_ip(mock_request)
    assert ip == "10.0.0.1"


# ---------------------------------------------------------------------------
# routers/health.py – cjpeg tool check (line 18) and jxl import fail (60-61)
# ---------------------------------------------------------------------------


def test_health_check_tools_with_cjpeg_encoder():
    """Cover the cjpeg entry in REQUIRED_TOOLS when JPEG_ENCODER=cjpeg."""
    from routers import health

    original_tools = health.REQUIRED_TOOLS.copy()
    try:
        health.REQUIRED_TOOLS["cjpeg"] = "cjpeg"
        tools = health.check_tools()
        assert "cjpeg" in tools
    finally:
        health.REQUIRED_TOOLS.clear()
        health.REQUIRED_TOOLS.update(original_tools)


def test_health_check_tools_jxl_import_failure():
    """Cover the ImportError fallback for jxl_plugin in check_tools."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("pillow_jxl", "jxlpy"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        from routers.health import check_tools

        tools = check_tools()
        assert tools["jxl_plugin"] is False


# ---------------------------------------------------------------------------
# optimizers/jpeg.py – CMYK/RGBA decode (line 89), ICC profile (111),
#   cap_quality binary search break (142)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_optimizer_cmyk_mode():
    """Cover JPEG _decode_image with CMYK mode conversion (line 89)."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()
    # Create CMYK JPEG
    img = Image.new("CMYK", (32, 32), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()

    config = OptimizationConfig(quality=60)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


@pytest.mark.asyncio
async def test_jpeg_optimizer_icc_profile_preservation():
    """Cover JPEG _pillow_encode with ICC profile (line 111)."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()

    # Create a JPEG with an ICC profile
    img = Image.new("RGB", (32, 32), (100, 150, 200))
    try:
        from PIL import ImageCms

        srgb = ImageCms.createProfile("sRGB")
        icc_data = ImageCms.ImageCmsProfile(srgb).tobytes()
    except Exception:
        icc_data = b"\x00" * 128

    buf = io.BytesIO()
    img.save(buf, format="JPEG", icc_profile=icc_data)
    data = buf.getvalue()

    config = OptimizationConfig(quality=60, strip_metadata=False)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


@pytest.mark.asyncio
async def test_jpeg_optimizer_max_reduction_cap():
    """Cover JPEG _cap_quality binary search (lines 130-154) and break at line 142."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()

    img = Image.new("RGB", (128, 128), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()

    config = OptimizationConfig(quality=30, max_reduction=10.0)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


@pytest.mark.asyncio
async def test_jpeg_optimizer_max_reduction_exceeds_cap():
    """Cover _cap_quality returning None when even q=100 exceeds cap."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()

    img = Image.new("RGB", (256, 256), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)
    data = buf.getvalue()

    config = OptimizationConfig(quality=10, max_reduction=0.01)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


# ---------------------------------------------------------------------------
# optimizers/jpeg.py – cjpeg legacy paths (lines 166-235)
# Uses mocked run_tool since cjpeg may not be installed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_optimizer_cjpeg_path():
    """Cover the cjpeg legacy pipeline (lines 166-235)."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()

    img = Image.new("RGB", (64, 64), (100, 150, 200))
    data = _jpeg_from_img(img, quality=90)
    small_jpeg = _jpeg_from_img(img, quality=50)

    async def mock_run_tool(cmd, input_data, **kwargs):
        return small_jpeg, b"", 0

    config = OptimizationConfig(quality=60)
    with patch("optimizers.jpeg.settings") as mock_settings:
        mock_settings.jpeg_encoder = "cjpeg"
        with patch("optimizers.jpeg.run_tool", side_effect=mock_run_tool):
            result = await opt._optimize_cjpeg(data, config)
            assert result.original_size == len(data)


@pytest.mark.asyncio
async def test_jpeg_optimizer_cjpeg_max_reduction():
    """Cover cjpeg _cap_mozjpeg binary search (lines 192-215)."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()

    img = Image.new("RGB", (128, 128), (200, 100, 50))
    data = _jpeg_from_img(img, quality=95)

    async def mock_run_tool(cmd, input_data, **kwargs):
        if "cjpeg" in cmd:
            q_idx = cmd.index("-quality") + 1 if "-quality" in cmd else -1
            q = int(cmd[q_idx]) if q_idx > 0 else 80
            ratio = q / 100.0
            size = int(len(data) * ratio)
            return data[:max(size, 100)], b"", 0
        return data, b"", 0

    config = OptimizationConfig(quality=30, max_reduction=5.0)
    with patch("optimizers.jpeg.settings") as mock_settings:
        mock_settings.jpeg_encoder = "cjpeg"
        with patch("optimizers.jpeg.run_tool", side_effect=mock_run_tool):
            result = await opt._optimize_cjpeg(data, config)
            assert result.original_size == len(data)


@pytest.mark.asyncio
async def test_jpeg_cjpeg_progressive():
    """Cover cjpeg progressive flag (line 233)."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()

    img = Image.new("RGB", (64, 64), (100, 150, 200))
    data = _jpeg_from_img(img, quality=90)

    called_with_progressive = [False]

    async def mock_run_tool(cmd, input_data, **kwargs):
        if "-progressive" in cmd:
            called_with_progressive[0] = True
        return data, b"", 0

    config = OptimizationConfig(quality=60, progressive_jpeg=True)
    with patch("optimizers.jpeg.settings") as mock_settings:
        mock_settings.jpeg_encoder = "cjpeg"
        with patch("optimizers.jpeg.run_tool", side_effect=mock_run_tool):
            result = await opt._optimize_cjpeg(data, config)
            assert called_with_progressive[0]


def test_jpeg_decode_to_bmp_rgba():
    """Cover _decode_to_bmp RGBA conversion (line 221)."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()

    img = Image.new("RGB", (32, 32), (100, 150, 200))
    data = _jpeg_from_img(img)

    with patch("optimizers.jpeg.Image.open") as mock_open:
        mock_open.return_value = Image.new("RGBA", (32, 32), (100, 150, 200, 255))
        result = opt._decode_to_bmp(data, True)
        assert isinstance(result, bytes)
        assert result[:2] == b"BM"


def test_jpeg_decode_to_bmp_unusual_mode():
    """Cover _decode_to_bmp unusual mode conversion (line 223)."""
    from optimizers.jpeg import JpegOptimizer

    opt = JpegOptimizer()

    img = Image.new("RGB", (32, 32), (100, 150, 200))
    data = _jpeg_from_img(img)

    with patch("optimizers.jpeg.Image.open") as mock_open:
        mock_open.return_value = Image.new("CMYK", (32, 32), (0, 0, 0, 0))
        result = opt._decode_to_bmp(data, True)
        assert isinstance(result, bytes)
        assert result[:2] == b"BM"


# ---------------------------------------------------------------------------
# optimizers/heic.py – _reencode path (lines 71-85)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heic_optimizer_reencode():
    """Cover HeicOptimizer._reencode (lines 66-85)."""
    import pillow_heif

    pillow_heif.register_heif_opener()

    from optimizers.heic import HeicOptimizer

    opt = HeicOptimizer()

    img = Image.new("RGB", (64, 64), (100, 150, 200))

    mock_heif_file = MagicMock()
    mock_heif_file.to_pillow.return_value = img

    with patch.object(pillow_heif, "open_heif", return_value=mock_heif_file):
        result = opt._reencode(b"\x00" * 100, quality=60)
        assert isinstance(result, bytes)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_heic_optimizer_reencode_with_icc():
    """Cover HeicOptimizer._reencode ICC profile preservation (line 82)."""
    import pillow_heif

    pillow_heif.register_heif_opener()

    from optimizers.heic import HeicOptimizer

    opt = HeicOptimizer()

    img = Image.new("RGB", (64, 64), (100, 150, 200))
    img.info["icc_profile"] = b"\x00" * 100

    mock_heif_file = MagicMock()
    mock_heif_file.to_pillow.return_value = img

    with patch.object(pillow_heif, "open_heif", return_value=mock_heif_file):
        result = opt._reencode(b"\x00" * 100, quality=40)
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# optimizers/jxl.py – full file coverage (lines 25-91)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jxl_optimizer_with_mock():
    """Cover JxlOptimizer _strip_metadata and _reencode."""
    from optimizers.jxl import JxlOptimizer

    opt = JxlOptimizer()

    original_data = b"\xff\x0a" + b"\x00" * 500
    small_output = b"\x00" * 100

    mock_jxlpy = MagicMock()
    with patch.dict("sys.modules", {"pillow_jxl": None, "jxlpy": mock_jxlpy}):
        with patch("optimizers.jxl.Image.open") as mock_open:
            mock_img = MagicMock(spec=Image.Image)
            mock_img.info = {}
            mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(small_output))
            mock_open.return_value = mock_img

            result_strip = opt._strip_metadata(original_data)
            assert isinstance(result_strip, bytes)
            assert len(result_strip) <= len(original_data)

            result_reencode = opt._reencode(original_data, quality=60)
            assert isinstance(result_reencode, bytes)


@pytest.mark.asyncio
async def test_jxl_optimizer_both_fail():
    """Cover JxlOptimizer fallback to 'none' when both methods fail."""
    from optimizers.jxl import JxlOptimizer

    opt = JxlOptimizer()
    data = b"\xff\x0a" + b"\x00" * 100

    with patch.object(opt, "_strip_metadata", side_effect=Exception("fail")):
        with patch.object(opt, "_reencode", side_effect=Exception("fail")):
            config = OptimizationConfig(quality=60, strip_metadata=True)
            result = await opt.optimize(data, config)
            assert result.method == "none"


@pytest.mark.asyncio
async def test_jxl_optimizer_strip_returns_original():
    """Cover JxlOptimizer._strip_metadata returning original when result is bigger."""
    from optimizers.jxl import JxlOptimizer

    opt = JxlOptimizer()

    small_data = b"\xff\x0a" + b"\x00" * 10

    mock_jxlpy = MagicMock()
    with patch.dict("sys.modules", {"pillow_jxl": None, "jxlpy": mock_jxlpy}):
        with patch("optimizers.jxl.Image.open") as mock_open:
            mock_img = MagicMock(spec=Image.Image)
            mock_img.info = {}
            mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"\x00" * 500))
            mock_open.return_value = mock_img

            result = opt._strip_metadata(small_data)
            assert result == small_data


# ---------------------------------------------------------------------------
# optimizers/avif.py – ICC profile lines (64, 90), strip returns original (71)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_avif_strip_returns_original_when_bigger():
    """Cover AvifOptimizer._strip_metadata returning original (line 71)."""
    from optimizers.avif import AvifOptimizer

    opt = AvifOptimizer()

    original_data = b"\x00" * 200

    with patch("optimizers.avif.Image.open") as mock_open:
        mock_img = MagicMock(spec=Image.Image)
        mock_img.info = {}
        mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"\x00" * 500))
        mock_open.return_value = mock_img

        mock_avif = MagicMock()
        with patch.dict("sys.modules", {"pillow_avif": mock_avif}):
            result = opt._strip_metadata(original_data)
            assert result == original_data


@pytest.mark.asyncio
async def test_avif_reencode_with_icc():
    """Cover AvifOptimizer._reencode ICC profile line (line 90)."""
    from optimizers.avif import AvifOptimizer

    opt = AvifOptimizer()

    with patch("optimizers.avif.Image.open") as mock_open:
        mock_img = MagicMock(spec=Image.Image)
        mock_img.info = {"icc_profile": b"\x00" * 50}
        mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"\x00" * 100))
        mock_open.return_value = mock_img

        mock_avif = MagicMock()
        with patch.dict("sys.modules", {"pillow_avif": mock_avif}):
            result = opt._reencode(b"\x00" * 200, quality=60)
            assert isinstance(result, bytes)
            call_kwargs = mock_img.save.call_args[1]
            assert "icc_profile" in call_kwargs


@pytest.mark.asyncio
async def test_avif_strip_with_icc():
    """Cover AvifOptimizer._strip_metadata ICC profile line (line 64)."""
    from optimizers.avif import AvifOptimizer

    opt = AvifOptimizer()

    with patch("optimizers.avif.Image.open") as mock_open:
        mock_img = MagicMock(spec=Image.Image)
        mock_img.info = {"icc_profile": b"\x00" * 50}
        mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"\x00" * 10))
        mock_open.return_value = mock_img

        mock_avif = MagicMock()
        with patch.dict("sys.modules", {"pillow_avif": mock_avif}):
            result = opt._strip_metadata(b"\x00" * 200)
            call_kwargs = mock_img.save.call_args[1]
            assert "icc_profile" in call_kwargs


# ---------------------------------------------------------------------------
# optimizers/bmp.py – RGBA opaque alpha (35-37), unusual mode (39),
#   palette None (106), odd-length RLE8 padding (189)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bmp_optimizer_rgba_opaque():
    """Cover BMP RGBA with fully opaque alpha (lines 34-37)."""
    from optimizers.bmp import BmpOptimizer

    opt = BmpOptimizer()

    img = Image.new("RGBA", (32, 32), (100, 150, 200, 255))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="BMP")
    data = buf.getvalue()

    with patch("optimizers.bmp.Image.open", return_value=img.copy()):
        config = OptimizationConfig(quality=60)
        result = await opt.optimize(data, config)
        assert result.original_size == len(data)


@pytest.mark.asyncio
async def test_bmp_optimizer_unusual_mode():
    """Cover BMP unusual mode conversion (line 39)."""
    from optimizers.bmp import BmpOptimizer

    opt = BmpOptimizer()

    img_i = Image.new("I", (32, 32))
    buf = io.BytesIO()
    img_i.convert("RGB").save(buf, format="BMP")
    data = buf.getvalue()

    with patch("optimizers.bmp.Image.open", return_value=img_i.copy()):
        config = OptimizationConfig(quality=60)
        result = await opt.optimize(data, config)
        assert result.original_size == len(data)


def test_bmp_rle8_null_palette():
    """Cover _encode_rle8_bmp returning None when palette is None (line 106)."""
    from optimizers.bmp import BmpOptimizer

    img = Image.new("P", (10, 10))
    img.putpalette([i % 256 for i in range(768)])
    with patch.object(img, "getpalette", return_value=None):
        result = BmpOptimizer._encode_rle8_bmp(img)
        assert result is None


def test_bmp_rle8_odd_literal_padding():
    """Cover RLE8 odd-length literal padding (line 197-198)."""
    from optimizers.bmp import _rle8_encode_row

    row = bytes([1, 2, 3])
    out = bytearray()
    _rle8_encode_row(row, out)

    data = bytes(out)
    assert len(data) > 0


def test_bmp_rle8_odd_literal_padding_5():
    """Cover RLE8 padding with a 5-byte literal run."""
    from optimizers.bmp import _rle8_encode_row

    row = bytes([10, 20, 30, 40, 50])
    out = bytearray()
    _rle8_encode_row(row, out)

    data = bytes(out)
    assert len(data) > 0


# ---------------------------------------------------------------------------
# optimizers/tiff.py – exif preservation (line 80), save exception (86-87)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tiff_optimizer_exif_preservation():
    """Cover TIFF _try_compression with exif and strip_metadata=False (line 80)."""
    from optimizers.tiff import TiffOptimizer

    opt = TiffOptimizer()

    img = Image.new("RGB", (32, 32), (100, 150, 200))
    buf = io.BytesIO()
    # Minimal valid EXIF structure
    exif_bytes = b"Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00"
    img.save(buf, format="TIFF", exif=exif_bytes)
    data = buf.getvalue()

    config = OptimizationConfig(quality=80, strip_metadata=False)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


@pytest.mark.asyncio
async def test_tiff_optimizer_compression_failure():
    """Cover TIFF _try_compression exception path (lines 86-87)."""
    from optimizers.tiff import TiffOptimizer

    opt = TiffOptimizer()

    # RGBA TIFF with quality < 70 triggers JPEG-in-TIFF which fails on RGBA
    img = Image.new("RGBA", (32, 32), (100, 150, 200, 128))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()

    config = OptimizationConfig(quality=40)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


# ---------------------------------------------------------------------------
# optimizers/webp.py – binary search (lines 74, 79)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webp_optimizer_max_reduction_binary_search():
    """Cover WebP _find_capped_quality binary search (lines 73-82)."""
    from optimizers.webp import WebpOptimizer

    opt = WebpOptimizer()

    img = Image.new("RGB", (128, 128), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=100)
    data = buf.getvalue()

    config = OptimizationConfig(quality=20, max_reduction=5.0)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


# ---------------------------------------------------------------------------
# optimizers/png.py – strip_metadata=False (line 39)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_optimizer_no_strip_metadata():
    """Cover PngOptimizer with strip_metadata=False (line 39)."""
    from optimizers.png import PngOptimizer

    opt = PngOptimizer()

    img = Image.new("RGB", (32, 32), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    config = OptimizationConfig(quality=60, strip_metadata=False)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


# ---------------------------------------------------------------------------
# security/ssrf.py – validate_url happy path (line 76)
# ---------------------------------------------------------------------------


def test_ssrf_validate_url_happy_path():
    """Cover validate_url returning the URL (line 76)."""
    from security.ssrf import validate_url

    with patch("security.ssrf.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]
        result = validate_url("https://example.com/image.png")
        assert result == "https://example.com/image.png"


# ---------------------------------------------------------------------------
# security/svg_sanitizer.py – _find_parent returns None (line 140)
# ---------------------------------------------------------------------------


def test_svg_sanitizer_find_parent_root():
    """Cover _find_parent returning None for root element (line 140)."""
    from security.svg_sanitizer import _find_parent
    from xml.etree.ElementTree import Element

    root = Element("svg")
    result = _find_parent(root, root)
    assert result is None


def test_svg_sanitizer_find_parent_not_found():
    """Cover _find_parent returning None when target not in tree."""
    from security.svg_sanitizer import _find_parent
    from xml.etree.ElementTree import Element

    root = Element("svg")
    child = Element("rect")
    root.append(child)
    orphan = Element("circle")

    result = _find_parent(root, orphan)
    assert result is None


# ---------------------------------------------------------------------------
# security/rate_limiter.py – burst limit call (line 115)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_burst_check():
    """Cover safe_check_rate_limit calling check_burst_limit (line 115)."""
    from security.rate_limiter import safe_check_rate_limit

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.redis_url = "redis://localhost:6379"
        with patch("security.rate_limiter.check_rate_limit", new_callable=AsyncMock):
            with patch(
                "security.rate_limiter.check_burst_limit", new_callable=AsyncMock
            ) as mock_bl:
                await safe_check_rate_limit("1.2.3.4", False)
                mock_bl.assert_awaited_once_with("1.2.3.4")


# ---------------------------------------------------------------------------
# utils/format_detect.py – JXL bare codestream (59), truncated PNG (130),
#   JXL ISO BMFF major (161), JXL compatible brand (180)
# ---------------------------------------------------------------------------


def test_format_detect_jxl_bare_codestream():
    """Cover JXL bare codestream detection (line 59)."""
    from utils.format_detect import detect_format

    data = b"\xff\x0a" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.JXL


def test_format_detect_jxl_isobmff_major():
    """Cover JXL ISO BMFF major brand detection (line 161)."""
    from utils.format_detect import detect_format

    box_size = struct.pack(">I", 20)
    ftyp = b"ftyp"
    major_brand = b"jxl "
    minor_version = b"\x00\x00\x00\x00"
    data = box_size + ftyp + major_brand + minor_version + b"\x00" * 100

    assert detect_format(data) == ImageFormat.JXL


def test_format_detect_jxl_isobmff_compat():
    """Cover JXL ISO BMFF compatible brand detection (line 180)."""
    from utils.format_detect import detect_format

    box_size = struct.pack(">I", 24)
    ftyp = b"ftyp"
    major_brand = b"unkn"
    minor_version = b"\x00\x00\x00\x00"
    compat_brand = b"jxl "
    data = box_size + ftyp + major_brand + minor_version + compat_brand + b"\x00" * 100

    assert detect_format(data) == ImageFormat.JXL


def test_format_detect_truncated_png():
    """Cover is_apng with truncated PNG (line 130 break)."""
    from utils.format_detect import is_apng

    png_sig = b"\x89PNG\r\n\x1a\n"
    chunk = struct.pack(">I", 1000) + b"IHDR" + b"\x00\x00"
    data = png_sig + chunk

    result = is_apng(data)
    assert result is False


# ---------------------------------------------------------------------------
# utils/metadata.py – truncated PNG chunk (lines 104-105)
# ---------------------------------------------------------------------------


def test_metadata_strip_truncated_png():
    """Cover _strip_png_metadata with truncated final chunk (lines 104-105)."""
    from utils.metadata import strip_metadata_selective

    img = Image.new("RGB", (8, 8), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    truncated = data[:-10]
    result = strip_metadata_selective(truncated, ImageFormat.PNG)
    assert isinstance(result, bytes)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# estimation/heuristics.py – _predict_jxl (lines 1030-1067)
# ---------------------------------------------------------------------------


def test_heuristics_predict_jxl_with_dimensions():
    """Cover _predict_jxl with valid dimensions (lines 1036-1046)."""
    from estimation.heuristics import _predict_jxl

    info = _make_header_info(
        ImageFormat.JXL,
        file_size=50000,
        dimensions={"width": 256, "height": 256},
    )

    config = OptimizationConfig(quality=40)
    result = _predict_jxl(info, config)
    assert result.confidence == "high"
    assert result.reduction_percent >= 0


def test_heuristics_predict_jxl_no_dimensions():
    """Cover _predict_jxl fallback with zero dimensions (lines 1047-1054)."""
    from estimation.heuristics import _predict_jxl

    info = _make_header_info(
        ImageFormat.JXL,
        file_size=50000,
        dimensions={"width": 0, "height": 0},
    )

    for quality in [40, 60, 80]:
        config = OptimizationConfig(quality=quality)
        result = _predict_jxl(info, config)
        assert result.confidence == "low"


def test_heuristics_predict_jxl_already_optimized():
    """Cover _predict_jxl when source_bpp <= target_bpp (line 1041-1042)."""
    from estimation.heuristics import _predict_jxl

    info = _make_header_info(
        ImageFormat.JXL,
        file_size=100,
        dimensions={"width": 256, "height": 256},
    )

    config = OptimizationConfig(quality=80)
    result = _predict_jxl(info, config)
    assert result.already_optimized is True


# ---------------------------------------------------------------------------
# estimation/heuristics.py – _predict_png lossless fallback (line 215),
#   gradient PNG bonus (280), low confidence (297)
# ---------------------------------------------------------------------------


def test_heuristics_predict_png_no_probes_no_cr():
    """Cover _predict_png_by_complexity returning 'low' confidence (line 297)."""
    from estimation.heuristics import _predict_png_by_complexity

    info = _make_header_info(
        ImageFormat.PNG,
        file_size=100000,
        dimensions={"width": 256, "height": 256},
        oxipng_probe_ratio=None,
        png_quantize_ratio=None,
        png_pngquant_probe_ratio=None,
        flat_pixel_ratio=None,
        unique_color_ratio=None,
    )

    config = OptimizationConfig(quality=60)
    reduction, potential, method, confidence = _predict_png_by_complexity(info, config)
    assert confidence == "low"


def test_heuristics_predict_png_lossless_fallback():
    """Cover _predict_png_by_complexity lossless_reduction = 5.0 fallback (line 215)."""
    from estimation.heuristics import _predict_png_by_complexity

    info = _make_header_info(
        ImageFormat.PNG,
        file_size=100000,
        dimensions={"width": 256, "height": 256},
        oxipng_probe_ratio=None,
        png_quantize_ratio=None,
        png_pngquant_probe_ratio=None,
        flat_pixel_ratio=None,
        unique_color_ratio=0.01,
    )

    config = OptimizationConfig(quality=80)
    reduction, potential, method, confidence = _predict_png_by_complexity(info, config)
    assert reduction >= 0


# ---------------------------------------------------------------------------
# estimation/heuristics.py – _predict_webp curve branches (584, 592)
# ---------------------------------------------------------------------------


def test_heuristics_webp_curve_80_high_delta():
    """Cover _curve_80 delta > 40 branch (line 584)."""
    from estimation.heuristics import _webp_interpolated_reduction

    result = _webp_interpolated_reduction(80, 50)
    assert result > 0


def test_heuristics_webp_curve_95_mid_delta():
    """Cover _curve_95 delta 15-35 branch (line 592)."""
    from estimation.heuristics import _webp_interpolated_reduction

    result = _webp_interpolated_reduction(95, 25)
    assert result > 0


def test_heuristics_bpp_to_quality_mid_range():
    """Cover _bpp_to_quality bpp 3.0-5.2 branch (line 530)."""
    from estimation.heuristics import _bpp_to_quality

    result = _bpp_to_quality(4.0)
    assert 80 <= result <= 95


# ---------------------------------------------------------------------------
# estimation/heuristics.py – GIF non-gradient palette bonus (668),
#   AVIF fallback (822-825), HEIC fallback (866, 870)
# ---------------------------------------------------------------------------


def test_heuristics_predict_gif_palette_bonus():
    """Cover GIF non-gradient palette reduction bonus (line 668)."""
    from estimation.heuristics import _predict_gif

    info = _make_header_info(
        ImageFormat.GIF,
        file_size=10000,
        dimensions={"width": 100, "height": 100},
        bit_depth=8,
        frame_count=1,
        unique_color_ratio=0.1,
        flat_pixel_ratio=0.5,
    )

    config = OptimizationConfig(quality=40)
    result = _predict_gif(info, config)
    assert result.reduction_percent >= 0


def test_heuristics_predict_avif_no_dimensions():
    """Cover AVIF fallback quality branches (lines 822-825)."""
    from estimation.heuristics import _predict_avif

    info = _make_header_info(
        ImageFormat.AVIF,
        file_size=50000,
        dimensions={"width": 0, "height": 0},
    )

    for quality in [40, 60, 80]:
        config = OptimizationConfig(quality=quality)
        result = _predict_avif(info, config)
        assert result.confidence == "low"


def test_heuristics_predict_heic_no_dimensions():
    """Cover HEIC fallback quality branches (lines 866, 870)."""
    from estimation.heuristics import _predict_heic

    info = _make_header_info(
        ImageFormat.HEIC,
        file_size=50000,
        dimensions={"width": 0, "height": 0},
    )

    config_high = OptimizationConfig(quality=40)
    result_high = _predict_heic(info, config_high)
    assert result_high.reduction_percent == 35.0

    config_low = OptimizationConfig(quality=80)
    result_low = _predict_heic(info, config_low)
    assert result_low.reduction_percent == 8.0


# ---------------------------------------------------------------------------
# estimation/heuristics.py – BMP photographic RLE bonus (line 986)
# ---------------------------------------------------------------------------


def test_heuristics_predict_bmp_photo_rle_bonus():
    """Cover BMP prediction RLE bonus for photographic content (line 986)."""
    from estimation.heuristics import _predict_bmp

    info = _make_header_info(
        ImageFormat.BMP,
        file_size=100000,
        dimensions={"width": 200, "height": 200},
        flat_pixel_ratio=0.2,
    )

    config = OptimizationConfig(quality=40)
    result = _predict_bmp(info, config)
    assert result.reduction_percent >= 0


# ---------------------------------------------------------------------------
# estimation/header_analysis.py – exception handlers, edge cases
# ---------------------------------------------------------------------------


def test_header_analysis_frame_count_exception():
    """Cover frame_count exception handler (lines 89-90)."""
    from estimation.header_analysis import analyze_header

    img = Image.new("RGB", (8, 8))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    data = buf.getvalue()

    with patch("estimation.header_analysis.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.size = (8, 8)
        mock_img.mode = "P"
        mock_img.info = {}
        type(mock_img).n_frames = property(lambda s: (_ for _ in ()).throw(Exception("fail")))
        mock_open.return_value = mock_img

        info = analyze_header(data, ImageFormat.GIF)
        assert info.frame_count == 1


def test_header_analysis_quantize_probe_exception():
    """Cover _quantize_probe exception handler (lines 334-335)."""
    from estimation.header_analysis import _quantize_probe

    mock_img = MagicMock(spec=Image.Image)
    mock_img.save = MagicMock(side_effect=Exception("bad image"))

    result = _quantize_probe(mock_img)
    assert result is None


# ---------------------------------------------------------------------------
# estimation/heuristics.py – _jpeg_probe and _png_lossy_probe exceptions
# ---------------------------------------------------------------------------


def test_heuristics_jpeg_probe_cmyk():
    """Cover _jpeg_probe CMYK conversion (line 441)."""
    from estimation.heuristics import _jpeg_probe

    img = Image.new("CMYK", (32, 32), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()

    result = _jpeg_probe(data, 60)
    # Should handle CMYK conversion without error


def test_heuristics_png_lossy_probe_exception():
    """Cover _png_lossy_probe exception handler (lines 513-514)."""
    from estimation.heuristics import _png_lossy_probe

    with patch("estimation.heuristics.subprocess.run", side_effect=Exception("fail")):
        result = _png_lossy_probe(b"\x00" * 100, 60)
        assert result is None


# ---------------------------------------------------------------------------
# estimation/header_analysis.py – JPEG quantization table parse failure
# ---------------------------------------------------------------------------


def test_header_analysis_jpeg_quality_parse_failure():
    """Cover _analyze_jpeg_extra exception handler (lines 368-369)."""
    from estimation.header_analysis import analyze_header

    jpeg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"

    info = analyze_header(jpeg_data, ImageFormat.JPEG)
    # Should not crash


# ---------------------------------------------------------------------------
# estimation/estimator.py – _thumbnail_compress (line 77)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimator_thumbnail_compress():
    """Cover estimation/estimator.py _thumbnail_compress (line 77)."""
    from estimation.estimator import _thumbnail_compress

    img = Image.new("RGB", (64, 64), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()

    result = await _thumbnail_compress(data, ImageFormat.JPEG, 60)
    if result is not None:
        assert 0 < result <= 1.5

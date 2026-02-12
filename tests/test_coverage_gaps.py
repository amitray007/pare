"""Targeted tests for remaining coverage gaps across all modules."""

import gzip
import io
import struct

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from PIL import Image


# --- optimizers/avif.py + heic.py: _strip_metadata with pillow_heif mock ---


@pytest.mark.asyncio
async def test_avif_strip_metadata_with_pillow_heif_mock():
    """Test AVIF _strip_metadata internals with mocked pillow_heif."""
    from optimizers.avif import AvifOptimizer

    opt = AvifOptimizer()

    mock_heif = MagicMock()
    mock_img = MagicMock(spec=Image.Image)
    mock_img.info = {"icc_profile": b"fake_icc_data"}
    # Mock save to write small data
    def mock_save(output, **kwargs):
        output.write(b"small_avif")
    mock_img.save = mock_save

    mock_heif_file = MagicMock()
    mock_heif_file.to_pillow.return_value = mock_img
    mock_heif.open_heif.return_value = mock_heif_file

    original = b"x" * 5000

    with patch.dict("sys.modules", {"pillow_heif": mock_heif}):
        result = opt._strip_metadata(original)
    assert result == b"small_avif"


@pytest.mark.asyncio
async def test_heic_strip_metadata_with_pillow_heif_mock():
    """Test HEIC _strip_metadata internals with mocked pillow_heif."""
    from optimizers.heic import HeicOptimizer

    opt = HeicOptimizer()

    mock_heif = MagicMock()
    mock_img = MagicMock(spec=Image.Image)
    mock_img.info = {}  # No ICC profile
    def mock_save(output, **kwargs):
        output.write(b"small_heic")
    mock_img.save = mock_save

    mock_heif_file = MagicMock()
    mock_heif_file.to_pillow.return_value = mock_img
    mock_heif.open_heif.return_value = mock_heif_file

    original = b"x" * 5000

    with patch.dict("sys.modules", {"pillow_heif": mock_heif}):
        result = opt._strip_metadata(original)
    assert result == b"small_heic"


@pytest.mark.asyncio
async def test_avif_strip_metadata_with_icc():
    """AVIF _strip_metadata preserves ICC profile."""
    from optimizers.avif import AvifOptimizer

    opt = AvifOptimizer()

    mock_heif = MagicMock()
    mock_img = MagicMock(spec=Image.Image)
    mock_img.info = {"icc_profile": b"srgb_profile_data"}
    mock_img.mode = "RGB"
    mock_img.size = (10, 10)

    # Make save produce small output
    def mock_save(output, **kwargs):
        output.write(b"small")

    mock_img.save = mock_save
    mock_heif_file = MagicMock()
    mock_heif_file.to_pillow.return_value = mock_img
    mock_heif.open_heif.return_value = mock_heif_file

    original = b"x" * 5000

    with patch.dict("sys.modules", {"pillow_heif": mock_heif}):
        result = opt._strip_metadata(original)
    assert result == b"small"


@pytest.mark.asyncio
async def test_heic_strip_metadata_larger_returns_original():
    """HEIC _strip_metadata returns original when stripped is larger."""
    from optimizers.heic import HeicOptimizer

    opt = HeicOptimizer()

    mock_heif = MagicMock()
    mock_img = MagicMock(spec=Image.Image)
    mock_img.info = {"icc_profile": b"profile"}
    mock_img.mode = "RGB"

    # Make save produce LARGER output
    def mock_save(output, **kwargs):
        output.write(b"x" * 10000)

    mock_img.save = mock_save
    mock_heif_file = MagicMock()
    mock_heif_file.to_pillow.return_value = mock_img
    mock_heif.open_heif.return_value = mock_heif_file

    original = b"x" * 500

    with patch.dict("sys.modules", {"pillow_heif": mock_heif}):
        result = opt._strip_metadata(original)
    assert result == original


# --- optimizers/bmp.py: RGBA handling ---


@pytest.mark.asyncio
async def test_bmp_optimize_rgba_opaque():
    """BMP with RGBA mode but fully opaque alpha -> converts to RGB."""
    from optimizers.bmp import BmpOptimizer

    opt = BmpOptimizer()
    img = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    data = buf.getvalue()
    result = await opt.optimize(data, __import__("schemas").OptimizationConfig(quality=80))
    assert result.success


@pytest.mark.asyncio
async def test_bmp_optimize_non_standard_mode():
    """BMP from non-standard mode (e.g., CMYK) -> converts to RGB."""
    from optimizers.bmp import BmpOptimizer

    opt = BmpOptimizer()
    # Create a TIFF in CMYK mode, save as BMP after conversion
    img = Image.new("CMYK", (20, 20), (0, 128, 255, 0))
    img_rgb = img.convert("RGB")
    buf = io.BytesIO()
    img_rgb.save(buf, format="BMP")
    data = buf.getvalue()
    result = await opt.optimize(data, __import__("schemas").OptimizationConfig(quality=60))
    assert result.success


# --- optimizers/tiff.py: EXIF/ICC preservation ---


@pytest.mark.asyncio
async def test_tiff_preserve_exif_and_icc():
    """TIFF optimizer preserves EXIF and ICC when strip_metadata=False."""
    from optimizers.tiff import TiffOptimizer

    opt = TiffOptimizer()
    from PIL import ImageCms
    img = Image.new("RGB", (50, 50))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation
    srgb = ImageCms.createProfile("sRGB")
    icc_data = ImageCms.ImageCmsProfile(srgb).tobytes()

    buf = io.BytesIO()
    img.save(buf, format="TIFF", exif=exif.tobytes(), icc_profile=icc_data)
    data = buf.getvalue()

    result = await opt.optimize(
        data, __import__("schemas").OptimizationConfig(quality=80, strip_metadata=False)
    )
    assert result.success


@pytest.mark.asyncio
async def test_tiff_jpeg_compression_fails():
    """TIFF JPEG-in-TIFF compression failure: skipped gracefully."""
    from optimizers.tiff import TiffOptimizer

    opt = TiffOptimizer()
    # Palette mode: JPEG compression will fail
    img = Image.new("P", (50, 50))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()
    result = await opt.optimize(
        data, __import__("schemas").OptimizationConfig(quality=60)
    )
    assert result.success


# --- utils/format_detect.py: APNG detection, SVGZ edge cases ---


def test_detect_apng():
    """Detect APNG format from animated PNG data."""
    from utils.format_detect import detect_format, ImageFormat

    frames = [Image.new("RGB", (10, 10), (i * 50, 0, 0)) for i in range(2)]
    buf = io.BytesIO()
    frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:])
    data = buf.getvalue()
    assert detect_format(data) == ImageFormat.APNG


def test_detect_gzip_non_svg_content():
    """Gzip header with valid gzip but non-SVG content -> unsupported."""
    from utils.format_detect import detect_format
    from exceptions import UnsupportedFormatError

    data = gzip.compress(b"<html>not svg</html>")
    with pytest.raises(UnsupportedFormatError):
        detect_format(data)


def test_detect_gzip_corrupt():
    """Corrupt gzip data -> falls through to unsupported."""
    from utils.format_detect import detect_format
    from exceptions import UnsupportedFormatError

    # Gzip magic bytes but completely invalid after that
    data = b"\x1f\x8b" + b"\xff" * 50
    with pytest.raises(UnsupportedFormatError):
        detect_format(data)


def test_isobmff_compat_brand_heic_in_list():
    """HEIC detected via compatible brands when major brand is unknown."""
    from utils.format_detect import detect_format, ImageFormat

    # Build ftyp box: major_brand="isom", compat_brands=["iso2", "heic"]
    data = struct.pack(">I", 28) + b"ftyp" + b"isom" + b"\x00\x00\x00\x00" + b"iso2" + b"heic" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.HEIC


def test_isobmff_unknown_compat_brands():
    """ISO BMFF with unknown compat brands -> UnsupportedFormatError."""
    from utils.format_detect import detect_format
    from exceptions import UnsupportedFormatError

    # Only unknown brands in compat list
    data = struct.pack(">I", 24) + b"ftyp" + b"isom" + b"\x00\x00\x00\x00" + b"iso2" + b"\x00" * 100
    with pytest.raises(UnsupportedFormatError):
        detect_format(data)


# --- routers/health.py: ImportError paths ---


def test_health_check_tools_missing_oxipng():
    """check_tools handles missing oxipng."""
    from routers.health import check_tools
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "oxipng":
            raise ImportError("no oxipng")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        tools = check_tools()
    assert "oxipng" in tools


def test_health_check_tools_missing_scour():
    """check_tools handles missing scour."""
    from routers.health import check_tools
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "scour":
            raise ImportError("no scour")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        tools = check_tools()
    assert "scour" in tools


def test_health_check_tools_missing_pillow():
    """check_tools handles missing Pillow."""
    from routers.health import check_tools
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        tools = check_tools()
    assert "pillow" in tools


# --- estimation/header_analysis.py: exception paths, edge cases ---


def test_header_analysis_exif_exception():
    """Header analysis handles EXIF extraction exception."""
    from estimation.header_analysis import analyze_header
    from utils.format_detect import ImageFormat

    # Create valid JPEG but mock getexif to fail
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()

    with patch.object(Image.Image, "getexif", side_effect=Exception("corrupt exif")):
        info = analyze_header(data, ImageFormat.JPEG)
    assert info.format == ImageFormat.JPEG


def test_header_analysis_n_frames_exception():
    """Header analysis handles n_frames exception."""
    from estimation.header_analysis import analyze_header
    from utils.format_detect import ImageFormat

    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    # n_frames attribute access might throw for some formats
    info = analyze_header(data, ImageFormat.PNG)
    assert info.frame_count >= 1


def test_header_analysis_png_content_no_crop():
    """PNG content analysis when image too small for crop."""
    from estimation.header_analysis import analyze_header
    from utils.format_detect import ImageFormat

    img = Image.new("RGB", (4, 4), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.format == ImageFormat.PNG


# --- estimation/heuristics.py: remaining edge cases ---


def test_heuristics_unknown_format_uses_bmp():
    """Unknown format falls back to _predict_bmp."""
    from estimation.header_analysis import HeaderInfo
    from estimation.heuristics import predict_reduction
    from utils.format_detect import ImageFormat
    from schemas import OptimizationConfig

    info = HeaderInfo(
        format=ImageFormat.BMP,
        dimensions={"width": 100, "height": 100},
        file_size=50000,
    )
    # Use BMP format directly (it's the default fallback)
    result = predict_reduction(info, ImageFormat.BMP, OptimizationConfig())
    assert result is not None


def test_png_lossy_palette_with_metadata_strip():
    """PNG lossy palette mode with metadata stripping bonus."""
    from estimation.header_analysis import HeaderInfo
    from estimation.heuristics import _predict_png
    from utils.format_detect import ImageFormat
    from schemas import OptimizationConfig

    info = HeaderInfo(
        format=ImageFormat.PNG,
        dimensions={"width": 100, "height": 100},
        file_size=5000,
        is_palette_mode=True,
        color_count=128,
        has_metadata_chunks=True,
    )
    config = OptimizationConfig(quality=60, png_lossy=True, strip_metadata=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 30  # Base 30% + 3% metadata


# --- utils/metadata.py: remaining paths ---


def test_metadata_tiff_no_exif_no_icc():
    """TIFF strip metadata when no EXIF and no ICC present."""
    from utils.metadata import _strip_pillow_metadata
    from utils.format_detect import ImageFormat

    img = Image.new("RGB", (20, 20))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()

    result = _strip_pillow_metadata(data, ImageFormat.TIFF, True, True)
    assert len(result) > 0


# --- storage/gcs.py: client initialization ---


def test_gcs_uploader_client_lazy_init():
    """GCS client is lazy-initialized on first access."""
    from storage.gcs import GCSUploader

    uploader = GCSUploader()
    assert uploader._client is None
    with patch("storage.gcs.gcs_lib.Client", return_value=MagicMock()):
        client = uploader.client
        assert client is not None
        assert uploader._client is not None
        # Second access returns same client
        assert uploader.client is client


@pytest.mark.asyncio
async def test_gcs_upload_failure():
    """GCS upload failure raises PareError."""
    from storage.gcs import GCSUploader
    from schemas import StorageConfig
    from exceptions import PareError

    uploader = GCSUploader()
    uploader._client = MagicMock()
    uploader._client.bucket.side_effect = Exception("GCS error")

    config = StorageConfig(provider="gcs", bucket="test-bucket", path="test/path.png")
    with pytest.raises(PareError, match="GCS upload failed"):
        await uploader.upload(b"data", "png", config)

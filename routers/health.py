import shutil

from fastapi import APIRouter

from config import settings
from schemas import HealthResponse

router = APIRouter()

REQUIRED_TOOLS = {
    "pngquant": "pngquant",
    "jpegtran": "jpegtran",
    "gifsicle": "gifsicle",
    "cwebp": "cwebp",
}
# cjpeg only required when using MozJPEG fallback
if settings.jpeg_encoder == "cjpeg":
    REQUIRED_TOOLS["cjpeg"] = "cjpeg"


def check_tools() -> dict[str, bool]:
    """Check availability of all required CLI compression tools."""
    results = {}
    for name, binary in REQUIRED_TOOLS.items():
        if shutil.which(binary):
            results[name] = True
        else:
            results[name] = False
    # Check Python libraries
    try:
        import oxipng  # noqa: F401

        results["oxipng"] = True
    except ImportError:
        results["oxipng"] = False
    try:
        import pillow_heif  # noqa: F401

        results["pillow_heif"] = True
    except ImportError:
        results["pillow_heif"] = False
    try:
        import scour  # noqa: F401

        results["scour"] = True
    except ImportError:
        results["scour"] = False
    try:
        from PIL import Image  # noqa: F401

        results["pillow"] = True
    except ImportError:
        results["pillow"] = False
    try:
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            import jxlpy  # noqa: F401
        results["jxl_plugin"] = True
    except ImportError:
        results["jxl_plugin"] = False
    return results


@router.get("/health", response_model=HealthResponse)
async def health():
    tools = check_tools()
    all_available = all(tools.values())
    return HealthResponse(
        status="ok" if all_available else "degraded",
        tools=tools,
        version="0.1.0",
    )

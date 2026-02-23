import shutil

from fastapi import APIRouter

from schemas import HealthResponse

router = APIRouter()


def check_tools() -> dict[str, bool]:
    """Check availability of all required tools and libraries."""
    results = {}

    # Core: pyvips (libvips)
    try:
        import pyvips

        results["pyvips"] = True
        # Verify key codecs are available
        results["jpegli"] = pyvips.type_find("VipsForeignSave", "jpegsave_buffer") != 0
        results["libheif"] = pyvips.type_find("VipsForeignSave", "heifsave_buffer") != 0
        results["libjxl"] = pyvips.type_find("VipsForeignSave", "jxlsave_buffer") != 0
        results["libwebp"] = pyvips.type_find("VipsForeignSave", "webpsave_buffer") != 0
    except ImportError:
        results["pyvips"] = False

    # CLI tools (only gifsicle remains)
    results["gifsicle"] = bool(shutil.which("gifsicle"))

    # Python libraries
    try:
        import oxipng  # noqa: F401

        results["oxipng"] = True
    except ImportError:
        results["oxipng"] = False
    try:
        import scour  # noqa: F401

        results["scour"] = True
    except ImportError:
        results["scour"] = False

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

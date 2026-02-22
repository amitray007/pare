import io
import json

from fastapi import APIRouter, File, Form, Request, UploadFile

from config import settings
from estimation.estimator import estimate as run_estimate
from estimation.presets import get_config_for_preset
from exceptions import BadRequestError, FileTooLargeError
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import detect_format
from utils.url_fetch import fetch_image

router = APIRouter()

LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB


@router.post("/estimate", response_model=EstimateResponse)
async def estimate(
    request: Request,
    file: UploadFile | None = File(None),
    options: str | None = Form(None),
    preset: str | None = Form(None),
):
    """Estimate compression savings without full optimization.

    Accepts multipart file upload or JSON with URL.
    Optionally accepts a preset (high/medium/low) instead of options.
    """
    content_type = request.headers.get("content-type", "")

    # --- Determine config from preset or options ---
    config = OptimizationConfig()
    if preset:
        try:
            config = get_config_for_preset(preset)
        except ValueError as e:
            raise BadRequestError(str(e))
    elif options:
        try:
            config = OptimizationConfig(**json.loads(options))
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Get image data ---
    if file is not None:
        data = await file.read()
    elif "application/json" in content_type:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise BadRequestError(f"Invalid JSON body: {e}")

        url = body.get("url")
        if not url:
            raise BadRequestError("Missing 'url' field in JSON body")

        # Use preset from JSON body if provided
        if not preset and body.get("preset"):
            try:
                config = get_config_for_preset(body["preset"])
            except ValueError as e:
                raise BadRequestError(str(e))

        is_authenticated = getattr(request.state, "is_authenticated", False)

        # Large-image thumbnail path
        thumbnail_url = body.get("thumbnail_url")
        client_file_size = body.get("file_size")

        if thumbnail_url and client_file_size and client_file_size >= LARGE_FILE_THRESHOLD:
            from estimation.estimator import estimate_from_thumbnail

            thumbnail_data = await fetch_image(thumbnail_url, is_authenticated=is_authenticated)
            detect_format(thumbnail_data)

            # Get original dimensions via Range request
            original_width, original_height = await _fetch_dimensions(
                url, is_authenticated
            )

            return await estimate_from_thumbnail(
                thumbnail_data=thumbnail_data,
                original_file_size=client_file_size,
                original_width=original_width,
                original_height=original_height,
                config=config,
            )

        data = await fetch_image(url, is_authenticated=is_authenticated)
    else:
        raise BadRequestError("Expected multipart/form-data or application/json")

    # Validate file size
    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File size {len(data)} bytes exceeds limit of {settings.max_file_size_mb} MB",
            file_size=len(data),
            limit=settings.max_file_size_bytes,
        )

    # Validate format
    detect_format(data)

    return await run_estimate(data, config)


async def _fetch_dimensions(url: str, is_authenticated: bool) -> tuple[int, int]:
    """Fetch just enough of the image to parse dimensions.

    Downloads first 8KB via Range request, parses with Pillow.
    Falls back to full download if Range not supported.
    """
    import httpx
    from PIL import Image

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={"Range": "bytes=0-8191"})
            partial = resp.content
            img = Image.open(io.BytesIO(partial))
            return img.size
    except Exception:
        # Fallback: download full image just for dimensions
        data = await fetch_image(url, is_authenticated=is_authenticated)
        img = Image.open(io.BytesIO(data))
        return img.size

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

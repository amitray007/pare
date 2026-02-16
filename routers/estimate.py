import json

from fastapi import APIRouter, File, Form, Request, UploadFile

from config import settings
from estimation.estimator import estimate as run_estimate
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
):
    """Estimate compression savings without compressing.

    Returns predicted size, reduction percent, confidence level,
    and the method that would be used.

    Estimation is lightweight (~20-50ms) and does not require
    a semaphore slot.
    """
    content_type = request.headers.get("content-type", "")

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

    # Parse optimization config from options field
    config = OptimizationConfig()
    if options:
        try:
            config = OptimizationConfig(**json.loads(options))
        except (json.JSONDecodeError, ValueError):
            pass

    # Run estimation (no semaphore needed — lightweight)
    return await run_estimate(data, config)

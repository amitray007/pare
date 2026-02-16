import json

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response

from config import settings
from exceptions import BadRequestError, FileTooLargeError
from optimizers.router import optimize_image
from schemas import (
    OptimizationConfig,
    OptimizeResponse,
    OptimizeResult,
    StorageConfig,
)
from storage.gcs import gcs_uploader
from utils.concurrency import compression_gate
from utils.format_detect import MIME_TYPES, detect_format
from utils.url_fetch import fetch_image

router = APIRouter()


@router.post("/optimize")
async def optimize(
    request: Request,
    file: UploadFile | None = File(None),
    options: str | None = Form(None),
):
    """Optimize an image.

    Two input modes:
    1. Multipart: file field + optional options JSON string
    2. JSON body: url field + inline optimization/storage config

    Two response modes:
    1. No storage config → raw bytes with X-* headers
    2. Storage config present → JSON with storage URL + stats
    """
    content_type = request.headers.get("content-type", "")

    if file is not None:
        # Multipart file upload mode
        data = await file.read()
        opt_config, storage_config = _parse_form_options(options)
    elif "application/json" in content_type:
        # JSON URL mode
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise BadRequestError(f"Invalid JSON body: {e}")
        url = body.get("url")
        if not url:
            raise BadRequestError("Missing 'url' field in JSON body")

        opt_config = OptimizationConfig(**body.get("optimization", {}))
        storage_config = StorageConfig(**body["storage"]) if "storage" in body else None

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

    # Validate format (will raise UnsupportedFormatError if unrecognized)
    detect_format(data)

    # Acquire compression slot (503 if queue full)
    await compression_gate.acquire()
    try:
        result = await optimize_image(data, opt_config)
    finally:
        compression_gate.release()

    # Response format depends on storage config
    if storage_config:
        return await _build_json_response(result, storage_config, request)
    else:
        return _build_binary_response(result, request)


def _parse_form_options(
    options_str: str | None,
) -> tuple[OptimizationConfig, StorageConfig | None]:
    """Parse the 'options' form field JSON string."""
    if not options_str:
        return OptimizationConfig(), None

    try:
        data = json.loads(options_str)
    except json.JSONDecodeError as e:
        raise BadRequestError(f"Invalid JSON in 'options' field: {e}")

    opt_config = OptimizationConfig(**data.get("optimization", {}))
    storage_config = None
    if "storage" in data:
        storage_config = StorageConfig(**data["storage"])

    return opt_config, storage_config


def _build_binary_response(result: OptimizeResult, request: Request) -> Response:
    """Build raw bytes response with X-* headers."""
    request_id = getattr(request.state, "request_id", "")

    return Response(
        content=result.optimized_bytes,
        media_type=MIME_TYPES.get(result.format, "application/octet-stream"),
        headers={
            "Content-Length": str(result.optimized_size),
            "X-Original-Size": str(result.original_size),
            "X-Optimized-Size": str(result.optimized_size),
            "X-Reduction-Percent": str(result.reduction_percent),
            "X-Original-Format": result.format,
            "X-Optimization-Method": result.method,
            "X-Request-ID": request_id,
        },
    )


async def _build_json_response(
    result: OptimizeResult,
    storage_config: StorageConfig,
    request: Request,
) -> dict:
    """Upload to storage and return JSON response."""
    storage_result = await gcs_uploader.upload(
        data=result.optimized_bytes,
        fmt=result.format,
        config=storage_config,
    )

    return OptimizeResponse(
        success=result.success,
        original_size=result.original_size,
        optimized_size=result.optimized_size,
        reduction_percent=result.reduction_percent,
        format=result.format,
        method=result.method,
        storage=storage_result,
        message=result.message,
    ).model_dump(exclude_none=True)

import io
import json

import httpx
from fastapi import APIRouter, File, Form, Request, UploadFile

from config import settings
from estimation.estimator import estimate as run_estimate
from estimation.estimator import estimate_from_header_bytes
from estimation.presets import get_config_for_preset
from exceptions import BadRequestError, FileTooLargeError
from routers.optimize import _read_upload_streaming
from schemas import EstimateResponse, OptimizationConfig
from utils.concurrency import estimate_gate
from utils.format_detect import ImageFormat, detect_format
from utils.image_validation import validate_image_dimensions
from utils.url_fetch import _get_client, fetch_image, fetch_partial

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
        # Multipart file upload mode -- streams in chunks to reject oversized
        # uploads before the full body is buffered in RAM.
        data = await _read_upload_streaming(file)

        # Validate file size
        if len(data) > settings.max_file_size_bytes:
            raise FileTooLargeError(
                f"File size exceeds limit of {settings.max_file_size_bytes} bytes",
                file_size=len(data),
                limit=settings.max_file_size_bytes,
            )

        # Validate format (raises UnsupportedFormatError if format is not recognized)
        fmt_detected = detect_format(data)

        # Validate decompressed size (reject images that would use too much memory)
        validate_image_dimensions(data)

        # Multipart header-only short-circuit: skip Pillow _open_image decode for
        # large PNG/JPEG when the fitted estimator is active.
        if settings.fitted_estimator_mode == "active":
            if (
                fmt_detected in (ImageFormat.PNG, ImageFormat.JPEG)
                and len(data) >= settings.header_only_min_size_bytes
            ):
                await estimate_gate.acquire()
                try:
                    result = await estimate_from_header_bytes(data, len(data), fmt_detected, config)
                    if result is not None:
                        return result
                finally:
                    estimate_gate.release()
                # Fall through to full estimation below

        # Acquire estimate slot (503 if queue full)
        await estimate_gate.acquire()
        try:
            return await run_estimate(data, config)
        finally:
            estimate_gate.release()

    elif "application/json" in content_type:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise BadRequestError(f"Invalid JSON body: {e}")

        url = body.get("url")
        if not url:
            raise BadRequestError("Missing 'url' field in JSON body")

        # Use preset or optimization config from JSON body
        if not preset and body.get("preset"):
            try:
                config = get_config_for_preset(body["preset"])
            except ValueError as e:
                raise BadRequestError(str(e))
        elif body.get("optimization"):
            config = OptimizationConfig(**body["optimization"])

        is_authenticated = getattr(request.state, "is_authenticated", False)

        # Large-image thumbnail path (unchanged — useful when caller has a real thumbnail)
        thumbnail_url = body.get("thumbnail_url")
        client_file_size = body.get("file_size")

        if (
            thumbnail_url
            and client_file_size
            and client_file_size >= settings.large_file_threshold_bytes
        ):
            from estimation.estimator import estimate_from_thumbnail

            thumbnail_data = await fetch_image(thumbnail_url, is_authenticated=is_authenticated)
            detect_format(thumbnail_data)

            # Get original dimensions via Range request
            original_width, original_height = await _fetch_dimensions(url, is_authenticated)

            return await estimate_from_thumbnail(
                thumbnail_data=thumbnail_data,
                original_file_size=client_file_size,
                original_width=original_width,
                original_height=original_height,
                config=config,
            )

        # NEW: Header-only fast path via Range fetch
        elif settings.fitted_estimator_mode == "active":
            data = None
            # Step 1: tiny range fetch to detect format AND get total size
            try:
                preview, total_size = await fetch_partial(
                    url, byte_range=(0, 7), is_authenticated=is_authenticated
                )
                fmt_detected = detect_format(preview)
            except Exception:
                # Fall through to full download
                fmt_detected = None
                total_size = None

            # Enforce max_file_size_bytes from the Range-fetch total before any full download
            if total_size is not None and total_size > settings.max_file_size_bytes:
                raise FileTooLargeError(
                    f"URL content too large: {total_size} bytes",
                    file_size=total_size,
                    limit=settings.max_file_size_bytes,
                )

            if (
                fmt_detected in (ImageFormat.PNG, ImageFormat.JPEG)
                and total_size is not None
                and total_size >= settings.header_only_min_size_bytes
            ):
                # Step 2: full header fetch
                header_bytes_needed = 100 if fmt_detected == ImageFormat.PNG else 8192
                try:
                    header_data, _ = await fetch_partial(
                        url,
                        byte_range=(0, header_bytes_needed - 1),
                        is_authenticated=is_authenticated,
                    )
                    result = await estimate_from_header_bytes(
                        header_data, total_size, fmt_detected, config
                    )
                    if result is not None:
                        return result
                except Exception:
                    pass  # Fall through to full download

            # Fall through: full download
            data = await fetch_image(url, is_authenticated=is_authenticated)

        else:
            data = await fetch_image(url, is_authenticated=is_authenticated)

        # Validate file size
        if len(data) > settings.max_file_size_bytes:
            raise FileTooLargeError(
                f"File size exceeds limit of {settings.max_file_size_bytes} bytes",
                file_size=len(data),
                limit=settings.max_file_size_bytes,
            )

        # Validate format
        detect_format(data)

        # Validate decompressed size (reject images that would use too much memory)
        validate_image_dimensions(data)

        # Acquire estimate slot (503 if queue full)
        await estimate_gate.acquire()
        try:
            return await run_estimate(data, config)
        finally:
            estimate_gate.release()

    else:
        raise BadRequestError("Expected multipart/form-data or application/json")


async def _fetch_dimensions(url: str, is_authenticated: bool) -> tuple[int, int]:
    """Fetch just enough of the image to parse dimensions.

    Downloads first 8KB via Range request, parses with Pillow.
    Falls back to full download if Range not supported.
    SSRF validation applied before any outbound request.
    Reuses the lifespan-managed pooled httpx client to avoid per-call
    TLS handshake overhead.
    """
    from PIL import Image

    from security.ssrf import validate_url

    validate_url(url)

    try:
        client = await _get_client()
        resp = await client.get(
            url,
            headers={"Range": "bytes=0-8191"},
            timeout=httpx.Timeout(10),
        )
        partial = resp.content
        img = Image.open(io.BytesIO(partial))
        return img.size
    except Exception:
        # Fallback: download full image just for dimensions
        data = await fetch_image(url, is_authenticated=is_authenticated)
        img = Image.open(io.BytesIO(data))
        return img.size

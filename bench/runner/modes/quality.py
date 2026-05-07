"""Quality mode: perceptual quality scoring for lossy optimizer outputs.

For each lossy case (jpeg, webp, avif, heic, jxl), runs optimize_image()
then scores the output with:
  - ssim          — pure-numpy single-scale SSIM, 0..1 (higher better)
  - psnr_db       — pure-numpy PSNR in dB; null for identical pixels
  - ssimulacra2   — subprocess; null if binary not on PATH
  - butteraugli_max / butteraugli_3norm — subprocess; null if binary missing

Lossless formats (png, apng, gif, bmp, tiff, svg, svgz) are filtered out
before iteration — quality metrics on lossless are trivially 1.0 / inf / 0
and provide no signal.

Output schema is a superset of quick mode's per-case dict:

    {
      ... quick-mode fields ...,
      "quality": {
        "ssim": 0.972,
        "psnr_db": 32.4,
        "ssimulacra2": 76.2,
        "butteraugli_max": 1.85,
        "butteraugli_3norm": 1.32,
        "wall_ms": 234.5,
        "perfect_match": false   # only present when True
      }
    }
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bench.runner.case import Case
from bench.runner.measure import measure
from bench.runner.modes.quick import measurement_to_dict
from bench.runner.probe import collect_tool_invocations
from bench.runner.quality import butteraugli_scores, psnr_db, ssim, ssimulacra2_score
from optimizers.router import optimize_image
from schemas import OptimizationConfig

logger = logging.getLogger(__name__)

# Formats that are always lossless (no quality degradation to measure).
# Trust case.fmt, not result.format or file extension.
_LOSSLESS_FORMATS: frozenset[str] = frozenset({"png", "apng", "gif", "bmp", "tiff", "svg", "svgz"})

# Lossy formats we score
_LOSSY_FORMATS: frozenset[str] = frozenset({"jpeg", "webp", "avif", "heic", "jxl"})


def _decode_to_array(data: bytes) -> np.ndarray:
    """Decode image bytes to a float32 numpy array in [0, 1], RGB."""
    import io

    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


async def _run_one_quality_case(case: Case, *, fast: bool = False) -> dict[str, Any]:
    input_data = case.load()
    config = OptimizationConfig(quality=case.quality)

    base: dict[str, Any] = {
        "case_id": case.case_id,
        "name": case.name,
        "bucket": case.bucket,
        "format": case.fmt,
        "preset": case.preset,
        "input_size": case.input_size,
        "iteration": 0,
    }

    # --- Optimize (timed via measure()) ---
    try:
        with measure() as m:
            with collect_tool_invocations() as invocations:
                result = await optimize_image(input_data, config)
    except Exception as exc:
        logger.warning("case %s optimize failed: %s", case.case_id, exc)
        return {
            **base,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # --- Decode both images ---
    try:
        ref_arr = await asyncio.to_thread(_decode_to_array, input_data)
        dist_arr = await asyncio.to_thread(_decode_to_array, result.optimized_bytes)
    except Exception as exc:
        logger.warning("case %s decode failed: %s", case.case_id, exc)
        return {
            **base,
            "measurement": measurement_to_dict(m),
            "tool_invocations": [
                {"tool": inv.tool, "wall_ms": inv.wall_ms, "exit_code": inv.exit_code}
                for inv in invocations
            ],
            "reduction_pct": result.reduction_percent,
            "method": result.method,
            "optimized_size": result.optimized_size,
            "quality": {
                "ssim": None,
                "psnr_db": None,
                "ssimulacra2": None,
                "butteraugli_max": None,
                "butteraugli_3norm": None,
                "wall_ms": 0.0,
                "decode_error": f"{type(exc).__name__}: {exc}",
            },
        }

    # --- Quality scoring ---
    q_start = time.perf_counter()

    # Shape guard: mismatched dimensions → null all metrics, log warning
    shapes_match = ref_arr.shape == dist_arr.shape
    if not shapes_match:
        logger.warning(
            "case %s: ref shape %s != dist shape %s; null metrics",
            case.case_id,
            ref_arr.shape,
            dist_arr.shape,
        )

    # Pure-numpy metrics (always attempted when shapes match)
    ssim_val: float | None = None
    psnr_val: float | None = None
    perfect_match = False

    if shapes_match:
        ssim_val = ssim(ref_arr, dist_arr)
        psnr_raw = psnr_db(ref_arr, dist_arr)
        if psnr_raw is None:
            perfect_match = True
        else:
            psnr_val = psnr_raw

    # Subprocess metrics via temp PNG files (skipped in fast mode)
    ss2_val: float | None = None
    ba_max: float | None = None
    ba_norm: float | None = None

    if not fast:
        with tempfile.TemporaryDirectory(prefix="pare_quality_") as tmpdir:
            tmp = Path(tmpdir)
            ref_png = tmp / "ref.png"
            dist_png = tmp / "dist.png"

            try:
                # Save decoded arrays back to lossless PNG so the tools see
                # exact pixel values (no re-encoding loss).
                ref_img = Image.fromarray((ref_arr * 255).astype(np.uint8), mode="RGB")
                dist_img = Image.fromarray((dist_arr * 255).astype(np.uint8), mode="RGB")
                ref_img.save(str(ref_png), format="PNG", compress_level=1)
                dist_img.save(str(dist_png), format="PNG", compress_level=1)

                ss2_val = ssimulacra2_score(ref_png, dist_png)
                ba_max, ba_norm = butteraugli_scores(ref_png, dist_png)
            except Exception as exc:
                logger.warning("case %s quality scoring error: %s", case.case_id, exc)

    q_wall_ms = (time.perf_counter() - q_start) * 1000.0

    quality_block: dict[str, Any] = {
        "ssim": round(ssim_val, 6) if ssim_val is not None else None,
        "psnr_db": round(psnr_val, 3) if psnr_val is not None else None,
        "ssimulacra2": round(ss2_val, 3) if ss2_val is not None else None,
        "butteraugli_max": round(ba_max, 4) if ba_max is not None else None,
        "butteraugli_3norm": round(ba_norm, 4) if ba_norm is not None else None,
        "wall_ms": round(q_wall_ms, 1),
    }
    if perfect_match:
        quality_block["perfect_match"] = True

    return {
        **base,
        "measurement": measurement_to_dict(m),
        "tool_invocations": [
            {"tool": inv.tool, "wall_ms": inv.wall_ms, "exit_code": inv.exit_code}
            for inv in invocations
        ],
        "reduction_pct": result.reduction_percent,
        "method": result.method,
        "optimized_size": result.optimized_size,
        "quality": quality_block,
    }


async def run_quality(cases: list[Case], *, fast: bool = False) -> list[dict[str, Any]]:
    """Sequentially run one optimize + quality score per lossy case.

    Lossless cases are filtered out before iteration. Sequential by design
    (matches quick/accuracy mode rationale: clean wall-time isolation).

    Args:
        cases: benchmark cases to run.
        fast: when True, skip SSIMULACRA2 + butteraugli subprocess calls and
              set those fields to null.  Reduces per-case wall time from ~3.5s
              to ~50ms.  SSIM and PSNR are always computed.
    """
    lossy_cases = [c for c in cases if c.fmt in _LOSSY_FORMATS]
    skipped = len(cases) - len(lossy_cases)
    if skipped:
        logger.info(
            "filtered %d lossless case(s) (only lossy formats scored in quality mode)",
            skipped,
        )

    results: list[dict[str, Any]] = []
    for case in lossy_cases:
        try:
            result = await _run_one_quality_case(case, fast=fast)
            results.append(result)
        except Exception as exc:
            logger.warning("case %s unexpected failure: %s", case.case_id, exc)
            results.append(
                {
                    "case_id": case.case_id,
                    "name": case.name,
                    "bucket": case.bucket,
                    "format": case.fmt,
                    "preset": case.preset,
                    "iteration": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return results


def run_quality_sync(cases: list[Case], *, fast: bool = False) -> list[dict[str, Any]]:
    """Synchronous wrapper for use from ``bench.runner.cli``.

    Args:
        cases: benchmark cases to run.
        fast: when True, skip SSIMULACRA2 + butteraugli subprocess calls.
    """
    return asyncio.run(run_quality(cases, fast=fast))

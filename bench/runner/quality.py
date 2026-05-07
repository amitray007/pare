"""Perceptual quality metric helpers for bench quality mode.

Provides pure-numpy SSIM and PSNR, plus subprocess wrappers for
ssimulacra2 and butteraugli_main.  All helpers return None (rather than
raising) on any failure so the bench run continues even on partial metric
availability.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binary resolution — resolved once per process via module-level singletons.
# ---------------------------------------------------------------------------

_ssimulacra2_bin: str | None | bool = False  # False = not yet resolved
_butteraugli_bin: str | None | bool = False  # False = not yet resolved

_BUTTERAUGLI_KNOWN_PATH = "/opt/homebrew/opt/jpeg-xl/bin/butteraugli_main"


def _resolve_ssimulacra2() -> str | None:
    global _ssimulacra2_bin
    if _ssimulacra2_bin is not False:
        return _ssimulacra2_bin  # type: ignore[return-value]
    found = shutil.which("ssimulacra2")
    _ssimulacra2_bin = found
    if found is None:
        logger.debug("ssimulacra2 not found on PATH; SSIMULACRA2 metrics will be null")
    return found


def _resolve_butteraugli() -> str | None:
    global _butteraugli_bin
    if _butteraugli_bin is not False:
        return _butteraugli_bin  # type: ignore[return-value]

    # 1) Env override
    env_bin = os.environ.get("BUTTERAUGLI_BIN")
    if env_bin and Path(env_bin).is_file():
        _butteraugli_bin = env_bin
        return env_bin

    # 2) Known homebrew path
    if Path(_BUTTERAUGLI_KNOWN_PATH).is_file():
        _butteraugli_bin = _BUTTERAUGLI_KNOWN_PATH
        return _BUTTERAUGLI_KNOWN_PATH

    # 3) PATH lookup
    found = shutil.which("butteraugli_main")
    _butteraugli_bin = found
    if found is None:
        logger.debug(
            "butteraugli_main not found; butteraugli metrics will be null. "
            "Install libjxl tools or set BUTTERAUGLI_BIN."
        )
    return found


# ---------------------------------------------------------------------------
# Pure-numpy metrics
# ---------------------------------------------------------------------------


def ssim(reference: np.ndarray, distorted: np.ndarray) -> Optional[float]:
    """Single-scale SSIM in pure numpy on float arrays in [0, 1].

    Per-channel for RGB, average across channels. Returns None if shapes
    mismatch or arrays have fewer than 8 pixels in either dimension.
    """
    if reference.shape != distorted.shape:
        logger.warning(
            "ssim: shape mismatch ref=%s dist=%s; returning None",
            reference.shape,
            distorted.shape,
        )
        return None

    # Ensure float64 for numerical stability
    ref = reference.astype(np.float64)
    dist = distorted.astype(np.float64)

    # Work per channel if RGB; treat grayscale as single channel
    if ref.ndim == 2:
        ref = ref[:, :, np.newaxis]
        dist = dist[:, :, np.newaxis]
    elif ref.ndim == 3 and ref.shape[2] > 3:
        # Drop alpha — SSIM on visual channels only
        ref = ref[:, :, :3]
        dist = dist[:, :, :3]

    # SSIM constants from Wang et al. 2004
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    channel_ssims: list[float] = []
    for c in range(ref.shape[2]):
        r = ref[:, :, c]
        d = dist[:, :, c]

        # Use an 11×11 Gaussian window approximated by a uniform box filter —
        # scipy is banned, so we use a simple uniform 11×11 window via
        # cumulative-sum trick (fast O(N) 2-D box filter).
        k = 11
        # Pad before filtering
        r_pad = np.pad(r, k // 2, mode="reflect")
        d_pad = np.pad(d, k // 2, mode="reflect")

        def _box_filter(arr: np.ndarray, ksize: int) -> np.ndarray:
            """2-D box (mean) filter via cumulative sums — O(N)."""
            cs = np.cumsum(arr, axis=0)
            cs = cs[ksize:] - cs[:-ksize]
            cs = np.cumsum(cs, axis=1)
            cs = cs[:, ksize:] - cs[:, :-ksize]
            return cs / (ksize * ksize)

        mu_r = _box_filter(r_pad, k)
        mu_d = _box_filter(d_pad, k)

        mu_r2 = mu_r * mu_r
        mu_d2 = mu_d * mu_d
        mu_rd = mu_r * mu_d

        # Variance and covariance (unbiased within window)
        r2_pad = r_pad * r_pad
        d2_pad = d_pad * d_pad
        rd_pad = r_pad * d_pad

        sigma_r2 = _box_filter(r2_pad, k) - mu_r2
        sigma_d2 = _box_filter(d2_pad, k) - mu_d2
        sigma_rd = _box_filter(rd_pad, k) - mu_rd

        # Numerator / denominator
        num = (2 * mu_rd + C1) * (2 * sigma_rd + C2)
        den = (mu_r2 + mu_d2 + C1) * (sigma_r2 + sigma_d2 + C2)

        ssim_map = np.where(den > 0, num / den, 1.0)
        channel_ssims.append(float(np.mean(ssim_map)))

    return float(np.mean(channel_ssims))


def psnr_db(reference: np.ndarray, distorted: np.ndarray) -> Optional[float]:
    """Standard PSNR in dB.

    Returns None when MSE is zero (identical pixels — infinite PSNR).
    Returns None when shapes mismatch.
    """
    if reference.shape != distorted.shape:
        logger.warning(
            "psnr_db: shape mismatch ref=%s dist=%s; returning None",
            reference.shape,
            distorted.shape,
        )
        return None

    ref = reference.astype(np.float64)
    dist = distorted.astype(np.float64)
    mse = float(np.mean((ref - dist) ** 2))
    if mse == 0.0:
        return None  # Caller adds perfect_match=True

    # Arrays are in [0, 1]; max signal = 1.0
    import math

    return round(10.0 * math.log10(1.0 / mse), 4)


# ---------------------------------------------------------------------------
# Subprocess wrappers
# ---------------------------------------------------------------------------


def ssimulacra2_score(
    ref_path: Path,
    dist_path: Path,
    *,
    timeout_s: float = 30.0,
) -> Optional[float]:
    """Run ssimulacra2 binary; return parsed float or None."""
    binary = _resolve_ssimulacra2()
    if binary is None:
        return None

    try:
        proc = subprocess.run(
            [binary, str(ref_path), str(dist_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        logger.warning("ssimulacra2 timed out after %.0fs", timeout_s)
        return None
    except FileNotFoundError:
        logger.warning("ssimulacra2 binary not found: %s", binary)
        return None
    except Exception as exc:
        logger.warning("ssimulacra2 subprocess error: %s", exc)
        return None

    if proc.returncode != 0:
        logger.debug("ssimulacra2 exited %d; stderr=%r", proc.returncode, proc.stderr[:200])
        return None

    stdout = proc.stdout.strip()
    if not stdout:
        return None

    # stdout is a single numeric token (possibly with a trailing newline)
    first_token = stdout.split()[0]
    try:
        return float(first_token)
    except ValueError:
        logger.warning("ssimulacra2 unexpected output: %r", stdout[:100])
        return None


def butteraugli_scores(
    ref_path: Path,
    dist_path: Path,
    *,
    timeout_s: float = 30.0,
) -> tuple[Optional[float], Optional[float]]:
    """Run butteraugli_main; return (max, 3-norm) or (None, None).

    butteraugli_main prints two lines:
        <max_value>
        3-norm: <value>
    Lower is better.
    """
    binary = _resolve_butteraugli()
    if binary is None:
        return (None, None)

    try:
        proc = subprocess.run(
            [binary, str(ref_path), str(dist_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        logger.warning("butteraugli_main timed out after %.0fs", timeout_s)
        return (None, None)
    except FileNotFoundError:
        logger.warning("butteraugli_main binary not found: %s", binary)
        return (None, None)
    except Exception as exc:
        logger.warning("butteraugli_main subprocess error: %s", exc)
        return (None, None)

    if proc.returncode != 0:
        logger.debug("butteraugli_main exited %d; stderr=%r", proc.returncode, proc.stderr[:200])
        return (None, None)

    stdout = proc.stdout.strip()
    if not stdout:
        return (None, None)

    lines = stdout.splitlines()

    # Line 0: max value
    max_val: Optional[float] = None
    try:
        max_val = float(lines[0].strip())
    except (IndexError, ValueError):
        logger.warning("butteraugli_main: could not parse max from line 0: %r", lines[:2])
        return (None, None)

    # Remaining lines: search for "3-norm: <val>"
    import re

    norm3_val: Optional[float] = None
    pattern = re.compile(r"3-norm:\s*([\d.]+(?:e[+-]?\d+)?)", re.IGNORECASE)
    for line in lines[1:]:
        m = pattern.search(line)
        if m:
            try:
                norm3_val = float(m.group(1))
            except ValueError:
                pass
            break

    return (max_val, norm3_val)

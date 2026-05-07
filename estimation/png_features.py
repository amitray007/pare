"""PNG image feature extraction for the fitted BPP estimator.

Features are computed from a 64×64 LANCZOS thumbnail of the decoded image.
All computations are NumPy-vectorized; no Python-level pixel loops.

Caller contract
---------------
``extract_png_features()`` is a **synchronous** function.  Callers from async context
must wrap it in ``asyncio.to_thread()`` per the project's async discipline.

The regression inference (loading ``PngModel``, applying ``StandardScaler``, evaluating
piecewise-linear coefficients) is implemented in ``estimation/estimator.py`` as
``_png_fitted_bpp()``, which calls this module to obtain its feature vector.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger("pare.estimation.png_features")

# Modes the model was trained on.  Images in other modes route to fallback.
_SUPPORTED_MODES: frozenset[str] = frozenset({"RGB", "RGBA", "L", "LA", "P"})

# Modes that carry an alpha channel (used to set has_alpha correctly for all supported modes).
_ALPHA_MODES: frozenset[str] = frozenset({"RGBA", "LA"})

# Thumbnail size for feature extraction.  Chosen to be large enough to capture spatial
# structure (Sobel gradient, color diversity) while remaining sub-millisecond to compute.
_THUMBNAIL_SIZE: int = 64

# Hard clips (spec §4).  Out-of-bounds → return None (caller routes to fallback).
MAX_UNIQUE_COLORS: int = 500_000
MAX_PIXELS: int = 100_000_000
MAX_INPUT_BPP: float = 64.0  # clearly above any real 8-bit RGBA (max = 32.0)

# Sobel magnitude threshold for edge_density classification (spec §4).
_SOBEL_EDGE_THRESHOLD: float = 32.0


@dataclass(frozen=True, slots=True)
class PngFeatures:
    """Feature vector for the PNG fitted-BPP regression model.

    All fields are the exact inputs the regression expects, in the order listed in
    the model artifact's ``features`` list.  Do not reorder without bumping ``model_version``.

    Attributes
    ----------
    has_alpha : bool
        True for modes with a real alpha channel (RGBA, LA) or palette images whose
        ``tRNS`` chunk indicates transparency.
    log10_unique_colors : float
        ``log10(unique_pixel_count)`` computed from the 64×64 thumbnail.
        Clipped at ``log10(MAX_UNIQUE_COLORS)`` before the regression to prevent
        out-of-training extrapolation.
    mean_sobel : float
        Mean of ``|∇I|`` (Sobel magnitude) on the grayscale projection of the thumbnail.
        Better-validated than Shannon entropy for compression prediction
        (Winkler/Yu QoMEX 2013; Larkin delentropy arXiv 1609.01117).
    edge_density : float
        Fraction of thumbnail pixels with Sobel magnitude > 32.0.  Disambiguates
        AA-text screenshots from photographs at similar ``unique_colors`` counts.
    quality : int
        Requested quality from ``OptimizationConfig`` (1–100).
    log10_orig_pixels : float
        ``log10(orig_w * orig_h)`` — the full-resolution pixel count, log-scaled.
        Clipped at ``log10(MAX_PIXELS)`` before the regression.
    input_bpp : float
        Input file size in bits-per-pixel: ``(orig_size * 8) / (orig_w * orig_h)``.
        Captures how heavily the source was already compressed before optimization.
        High-BPP inputs (raw/lossless PNGs) compress much more than low-BPP ones.
        Clipped at ``MAX_INPUT_BPP`` before the regression.
    """

    has_alpha: bool
    log10_unique_colors: float
    mean_sobel: float
    edge_density: float
    quality: int
    log10_orig_pixels: float
    input_bpp: float


def extract_png_features(
    img: Image.Image,
    orig_w: int,
    orig_h: int,
    quality: int,
    orig_size: int = 0,
) -> PngFeatures | None:
    """Extract regression features from a decoded PNG image.

    Parameters
    ----------
    img :
        Decoded Pillow image.  Must be in a supported mode (RGB, RGBA, L, LA, P).
        Images in unsupported modes return ``None``.
    orig_w, orig_h :
        Full-resolution width and height (pixels).  Used to compute ``log10_orig_pixels``.
        If ``orig_w * orig_h`` exceeds ``MAX_PIXELS``, returns ``None``.
    quality :
        Requested optimization quality (1–100), passed through as a feature.
    orig_size :
        Original file size in bytes.  Used to compute ``input_bpp``.  Pass 0 to
        disable the ``input_bpp`` clip check (feature will be 0.0, which the model
        was not trained on — callers should always pass the real size).

    Returns
    -------
    PngFeatures
        Extracted feature vector, or ``None`` if the image is unsupported,
        pixel count exceeds the hard clip, or ``input_bpp`` exceeds ``MAX_INPUT_BPP``.

    Notes
    -----
    - For palette mode (``P``) with ``tRNS``, the count is (unique colors observed in
      the 64×64 thumbnail) + 1 — accounting for the transparent palette index
      without requiring a full-image color enumeration.
    - The 64×64 thumbnail is computed with LANCZOS resampling for best quality
      at the small target size.
    - All NumPy work operates on contiguous arrays; no Python pixel loops.
    """
    if img.mode not in _SUPPORTED_MODES:
        return None

    orig_pixels = orig_w * orig_h
    if orig_pixels > MAX_PIXELS:
        return None

    # --- has_alpha ---
    # RGBA and LA have explicit alpha channels.
    # Palette images are transparent if tRNS is present in img.info.
    if img.mode in _ALPHA_MODES:
        has_alpha = True
    elif img.mode == "P":
        has_alpha = "transparency" in img.info
    else:
        has_alpha = False

    # --- Thumbnail ---
    thumb = img.resize((_THUMBNAIL_SIZE, _THUMBNAIL_SIZE), Image.LANCZOS)

    # --- log10_unique_colors ---
    # Convert to a consistent array form for unique-row counting.
    # For palette mode, convert to RGBA (if transparent) or RGB first so the pixel
    # values are actual colors, not palette indices.
    if thumb.mode == "P":
        # has_alpha already set above; palette_size before conversion.
        palette_size = len(thumb.getpalette()) // 3  # each entry is (R, G, B)
        thumb_for_count = thumb.convert("RGBA" if has_alpha else "RGB")
        arr = np.asarray(thumb_for_count)
        # channels dimension
        h, w, channels = arr.shape
        unique_count = np.unique(arr.reshape(-1, channels), axis=0).shape[0]
        if has_alpha:
            # One extra pseudo-color for the transparent index
            unique_count = min(unique_count + 1, palette_size + 1)
    elif thumb.mode in ("L", "LA"):
        arr = np.asarray(thumb)
        if thumb.mode == "LA":
            h, w, channels = arr.shape
        else:
            # L is 2D; reshape to (N, 1) for consistent unique-row counting
            channels = 1
            arr = arr.reshape(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE, 1)
            h, w = _THUMBNAIL_SIZE, _THUMBNAIL_SIZE
        unique_count = np.unique(arr.reshape(-1, channels), axis=0).shape[0]
    else:
        # RGB or RGBA
        arr = np.asarray(thumb)
        h, w, channels = arr.shape
        unique_count = np.unique(arr.reshape(-1, channels), axis=0).shape[0]

    if unique_count > MAX_UNIQUE_COLORS:
        return None

    log10_unique_colors = math.log10(max(unique_count, 1))

    # --- Grayscale projection for Sobel ---
    # Convert thumbnail to grayscale float for gradient computation.
    gray = np.asarray(thumb.convert("L"), dtype=np.float32)

    # scipy.ndimage.sobel operates on one axis at a time; magnitude = hypot(Sx, Sy).
    from scipy.ndimage import sobel as _sobel

    sx = _sobel(gray, axis=1)  # horizontal gradient
    sy = _sobel(gray, axis=0)  # vertical gradient
    sobel_mag = np.hypot(sx, sy)

    mean_sobel = float(sobel_mag.mean())
    edge_density = float((sobel_mag > _SOBEL_EDGE_THRESHOLD).mean())

    # --- log10_orig_pixels ---
    log10_orig_pixels = math.log10(max(orig_pixels, 1))

    # --- input_bpp ---
    # Bits-per-pixel of the *input* file (not the decoded image).
    # High-BPP sources (raw/lossless PNGs) compress much more than low-BPP ones.
    if orig_size > 0:
        input_bpp = (orig_size * 8) / orig_pixels
        if input_bpp > MAX_INPUT_BPP:
            return None  # feature_oob — clearly above any real 8-bit RGBA
    else:
        input_bpp = 0.0  # caller did not provide size; feature disabled

    return PngFeatures(
        has_alpha=has_alpha,
        log10_unique_colors=log10_unique_colors,
        mean_sobel=mean_sobel,
        edge_density=edge_density,
        quality=quality,
        log10_orig_pixels=log10_orig_pixels,
        input_bpp=input_bpp,
    )

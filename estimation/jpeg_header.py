"""Pure-bytes JPEG header parser + LSM source-quality estimation.

Parses SOI/SOF/DQT/SOS/APP14 markers from a JPEG byte stream without
Pillow — only ``memoryview`` + ``int.from_bytes``.  No ``struct.unpack``.

Security hardening
------------------
- ``MAX_HEAD_BYTES = 65_536`` — never read past this offset.
- ``MAX_MARKERS = 256`` — bounded marker-walk iteration.
- ``MAX_DQT_BYTES = 4096`` — cap total bytes consumed by DQT segments.
- Every variable-length segment undergoes an underflow check (``seg_len >= 2``)
  and a slice-bounds guard (``i + seg_len <= n``) before any access.
- ``Nf == 0``, ``width == 0``, and ``height == 0`` are rejected immediately.
- On any structural / validation failure: return ``None``.  **Never raise.**

Atheris fuzzing planned for Phase 2 verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------
MAX_HEAD_BYTES: int = 65_536
MAX_MARKERS: int = 256
MAX_DQT_BYTES: int = 4096

# ---------------------------------------------------------------------------
# JPEG Annex K reference tables (quality 50 baseline)
#
# The JPEG standard (Annex K) specifies tables in natural 8×8 raster order,
# but DQT segments in JPEG files store coefficients in zigzag scan order.
# These constants are stored in zigzag order so they can be compared directly
# against the parsed DQT bytes without any reordering.
#
# Zigzag conversion from natural order:
#   JPEG_ZIGZAG[i] gives the natural-order position that appears at zigzag
#   index i.  The arrays below have been pre-converted.
# ---------------------------------------------------------------------------

# fmt: off
ANNEX_K_LUMA = np.array([
    16, 11, 12, 14, 12, 10, 16, 14,
    13, 14, 18, 17, 16, 19, 24, 40,
    26, 24, 22, 22, 24, 49, 35, 37,
    29, 40, 58, 51, 61, 60, 57, 51,
    56, 55, 64, 72, 92, 78, 64, 68,
    87, 69, 55, 56, 80, 109, 81, 87,
    95, 98, 103, 104, 103, 62, 77, 113,
    121, 112, 100, 120, 92, 101, 103, 99,
], dtype=np.int32)

ANNEX_K_CHROMA = np.array([
    17, 18, 18, 24, 21, 24, 47, 26,
    26, 47, 99, 66, 56, 66, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
], dtype=np.int32)
# fmt: on


# ---------------------------------------------------------------------------
# JPEG marker constants
# ---------------------------------------------------------------------------
_SOI = 0xFFD8
_SOF0 = 0xFFC0
_SOF1 = 0xFFC1
_SOF2 = 0xFFC2  # progressive
_SOF3 = 0xFFC3  # lossless
_DHT = 0xFFC4
_RST_MIN = 0xFFD0
_RST_MAX = 0xFFD7
_SOI_MARK = 0xFFD8
_EOI = 0xFFD9
_SOS = 0xFFDA
_DQT = 0xFFDB
_APP0 = 0xFFE0
_APP14 = 0xFFEE
_COM = 0xFFFE


@dataclass(frozen=True, slots=True)
class JpegHeader:
    """Parsed JPEG header fields.

    Attributes
    ----------
    width, height : int
        Image dimensions. Both are > 0.
    components : int
        Nf from SOF — 1 (grayscale) or 3 (color).
    bit_depth : int
        Sample precision from SOF — typically 8.
    subsampling : str
        One of ``"4:4:4"``, ``"4:2:2"``, ``"4:2:0"``, ``"grayscale"``, ``"other"``.
    progressive : bool
        True when SOF2 marker was present.
    dqt_luma : list[int]
        64-element zigzag-order quantization table (Tq=0).
    dqt_chroma : list[int] | None
        64-element Tq=1 table, or None for grayscale or missing.
    app14_color_transform : int | None
        APP14 Adobe color transform byte: 0=Unknown/RGB, 1=YCbCr, 2=YCCK.
        None if no APP14 marker was seen.
    fallback_reason : str | None
        Set when the JPEG was parsed successfully but is known to be
        unfit for modelling.  Possible values:

        - ``"lossless_jpeg"`` — SOF3 marker.
        - ``"non_standard_components"`` — Nf not in {1, 3}.
        - ``"non_default_color_transform"`` — APP14 color_transform ∉ {None, 0, 1}.
        - ``"missing_chroma_table"`` — Nf==3 but no Tq=1 DQT found.

        The caller decides whether to honour the reason.
    """

    width: int
    height: int
    components: int
    bit_depth: int
    subsampling: str
    progressive: bool
    dqt_luma: list[int]
    dqt_chroma: list[int] | None
    app14_color_transform: int | None
    fallback_reason: str | None


def _derive_subsampling(
    components: int,
    sampling_factors: list[tuple[int, int]],
) -> str:
    """Derive subsampling string from sampling factors.

    Parameters
    ----------
    components :
        Number of image components (Nf).
    sampling_factors :
        List of (Hi, Vi) tuples, one per component.
    """
    if components == 1:
        return "grayscale"
    if len(sampling_factors) < 3:
        return "other"
    h0, v0 = sampling_factors[0]
    h1, v1 = sampling_factors[1]
    h2, v2 = sampling_factors[2]
    if (h0, v0, h1, v1, h2, v2) == (1, 1, 1, 1, 1, 1):
        return "4:4:4"
    if (h0, v0, h1, v1, h2, v2) == (2, 1, 1, 1, 1, 1):
        return "4:2:2"
    if (h0, v0, h1, v1, h2, v2) == (2, 2, 1, 1, 1, 1):
        return "4:2:0"
    return "other"


def parse_jpeg_header(data: bytes) -> JpegHeader | None:
    """Parse JPEG header from *data*.

    Returns a ``JpegHeader`` on success (which may have a non-None
    ``fallback_reason``), or ``None`` on hard structural failure.

    Only the first ``MAX_HEAD_BYTES`` bytes are examined.  Walks at most
    ``MAX_MARKERS`` variable-length segments and consumes at most
    ``MAX_DQT_BYTES`` total across all DQT segments.

    Never raises on malformed input.
    """
    if len(data) < 4:
        return None

    # Clamp to security window
    n = min(len(data), MAX_HEAD_BYTES)
    mv = memoryview(data[:n])

    # Check SOI marker
    if int.from_bytes(mv[0:2], "big") != _SOI_MARK:
        return None

    # State
    width: int = 0
    height: int = 0
    components: int = 0
    bit_depth: int = 8
    sampling_factors: list[tuple[int, int]] = []
    progressive: bool = False
    dqt_tables: dict[int, list[int]] = {}  # Tq → 64-element list
    app14_color_transform: int | None = None
    fallback_reason: str | None = None
    sof_seen: bool = False
    dqt_bytes_consumed: int = 0

    i = 2  # position after SOI
    marker_count = 0

    while i + 1 < n and marker_count < MAX_MARKERS:
        # Skip fill bytes (0xFF padding between markers is legal)
        while i < n and mv[i] == 0xFF:
            i += 1
        if i >= n:
            break

        # At this point mv[i] is the second byte of a marker (0xFF already consumed)
        marker_byte = mv[i]
        i += 1
        marker_count += 1
        marker = 0xFF00 | marker_byte

        # Stand-alone markers (no length field)
        if marker == _SOI_MARK or (_RST_MIN <= marker <= _RST_MAX) or marker == _EOI:
            if marker == _EOI:
                break
            continue

        # SOS — end of headers; stop walking
        if marker == _SOS:
            break

        # All other markers: 2-byte length field follows
        if i + 2 > n:
            break
        seg_len = int.from_bytes(mv[i : i + 2], "big")
        if seg_len < 2:
            # Underflow: length must include its own 2 bytes
            return None
        if i + seg_len > n:
            # Slice-bounds guard: segment extends beyond our window
            break

        seg = mv[i + 2 : i + seg_len]  # payload (seg_len - 2 bytes)

        # --- SOF markers ---
        if marker in (_SOF0, _SOF1, _SOF2, _SOF3):
            if marker == _SOF3:
                fallback_reason = "lossless_jpeg"
                # Still capture dimensions/components if parseable
                if len(seg) >= 6:
                    bit_depth = seg[0]
                    height = int.from_bytes(seg[1:3], "big")
                    width = int.from_bytes(seg[3:5], "big")
                    components = seg[5]
                    sof_seen = True
                # Return immediately — caller routes to fallback
                i += seg_len
                break

            if len(seg) < 6:
                return None
            bit_depth = seg[0]
            height = int.from_bytes(seg[1:3], "big")
            width = int.from_bytes(seg[3:5], "big")
            components = seg[5]

            if width == 0 or height == 0:
                return None
            if components == 0:
                return None

            progressive = marker == _SOF2
            sof_seen = True

            # Parse per-component sampling factors
            sampling_factors = []
            for c in range(components):
                off = 6 + c * 3
                if off + 2 >= len(seg):
                    break
                sampling_byte = seg[off + 1]
                hi = (sampling_byte >> 4) & 0x0F
                vi = sampling_byte & 0x0F
                sampling_factors.append((hi, vi))

        # --- DQT markers ---
        elif marker == _DQT:
            seg_payload = bytes(seg)
            pos = 0
            seg_payload_len = len(seg_payload)
            while pos < seg_payload_len:
                if pos + 1 > seg_payload_len:
                    break
                dqt_info = seg_payload[pos]
                pos += 1
                precision = (dqt_info >> 4) & 0x0F  # 0=8-bit, 1=16-bit
                tq = dqt_info & 0x0F
                table_bytes = 64 * (2 if precision == 1 else 1)
                if pos + table_bytes > seg_payload_len:
                    break
                dqt_bytes_consumed += table_bytes
                if dqt_bytes_consumed > MAX_DQT_BYTES:
                    break
                if precision == 0:
                    table = [seg_payload[pos + k] for k in range(64)]
                else:
                    table = [
                        int.from_bytes(seg_payload[pos + k * 2 : pos + k * 2 + 2], "big")
                        for k in range(64)
                    ]
                dqt_tables[tq] = table
                pos += table_bytes

        # --- APP14 (Adobe) ---
        elif marker == _APP14:
            # APP14 layout: identifier(5) + version(2) + flags0(2) + flags1(2) + transform(1)
            seg_bytes = bytes(seg)
            if len(seg_bytes) >= 12 and seg_bytes[:5] == b"Adobe":
                app14_color_transform = seg_bytes[11]

        i += seg_len

    # --- Post-walk validation ---
    if not sof_seen:
        return None

    if fallback_reason == "lossless_jpeg":
        # Partial parse — return with reason so caller can route
        dqt_luma = dqt_tables.get(0, [])
        return JpegHeader(
            width=width,
            height=height,
            components=components,
            bit_depth=bit_depth,
            subsampling="other",
            progressive=progressive,
            dqt_luma=dqt_luma,
            dqt_chroma=dqt_tables.get(1),
            app14_color_transform=app14_color_transform,
            fallback_reason=fallback_reason,
        )

    if components not in (1, 3):
        fallback_reason = "non_standard_components"

    if app14_color_transform is not None and app14_color_transform not in (0, 1):
        fallback_reason = "non_default_color_transform"

    dqt_luma = dqt_tables.get(0, [])
    dqt_chroma = dqt_tables.get(1)

    if fallback_reason is None and components == 3 and dqt_chroma is None:
        fallback_reason = "missing_chroma_table"

    subsampling = _derive_subsampling(components, sampling_factors)

    return JpegHeader(
        width=width,
        height=height,
        components=components,
        bit_depth=bit_depth,
        subsampling=subsampling,
        progressive=progressive,
        dqt_luma=dqt_luma,
        dqt_chroma=dqt_chroma,
        app14_color_transform=app14_color_transform,
        fallback_reason=fallback_reason,
    )


# ---------------------------------------------------------------------------
# LSM source-quality estimation
# ---------------------------------------------------------------------------


def estimate_source_quality_lsm(
    dqt_luma: list[int],
    dqt_chroma: list[int] | None,
) -> tuple[int, float]:
    """Estimate the encoder's quality setting by matching DQT against Annex K tables.

    Uses NumPy broadcast over all 100 candidate quality levels — no Python loop
    over Q.

    Parameters
    ----------
    dqt_luma :
        64-element zigzag-order luma quantization table.
    dqt_chroma :
        64-element chroma quantization table, or None for grayscale.

    Returns
    -------
    q_source : int
        Estimated source quality, in [1, 100].
    nse : float
        Normalised Sum-Squared Error, in [0, 1].  Values < 0.85 indicate
        custom quantization tables that don't match Annex K scaling — the
        caller should treat these as a fallback case.
    """
    obs_l = np.array(dqt_luma, dtype=np.int32)

    Q = np.arange(1, 101, dtype=np.int32)  # (100,)
    S = np.where(Q < 50, 5000 // Q, 200 - 2 * Q).reshape(-1, 1)  # (100, 1)

    T_lum = np.clip((S * ANNEX_K_LUMA + 50) // 100, 1, 255).astype(np.int32)  # (100, 64)
    sse = ((T_lum - obs_l) ** 2).sum(axis=1)  # (100,)

    if dqt_chroma is not None:
        obs_c = np.array(dqt_chroma, dtype=np.int32)
        T_chrom = np.clip((S * ANNEX_K_CHROMA + 50) // 100, 1, 255).astype(np.int32)
        sse = sse + ((T_chrom - obs_c) ** 2).sum(axis=1)

    q_source = int(np.argmin(sse)) + 1
    sse_min = float(sse.min())

    # NSE = 1 - SSE_min / SSE_baseline; baseline = variance of observed values
    obs_full = np.concatenate(
        [obs_l] + ([np.array(dqt_chroma, dtype=np.int32)] if dqt_chroma is not None else [])
    )
    baseline = float(((obs_full - obs_full.mean()) ** 2).sum())
    nse = 1.0 - sse_min / baseline if baseline > 0 else 0.0
    nse = max(0.0, min(1.0, nse))
    return q_source, nse

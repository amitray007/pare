"""Benchmark case definitions.

Every raster format gets the same 3 core sizes (small/medium/large) for
consistent coverage of both exact and sample estimation paths.  Lossy photo
formats additionally vary source quality (95/75/50).

SVG/SVGZ have 3 fixed complexity variants.

Also supports loading real-world image files from a corpus directory
via load_corpus_cases().
"""

from dataclasses import dataclass
from pathlib import Path

from benchmarks.constants import CORE_SIZES, LOSSY_QUALITIES
from benchmarks.generators import (
    encode_image,
    gradient,
    graphic_like,
    photo_like,
    screenshot_like,
    svg_bloated,
    svg_complex,
    svg_simple,
    svgz_from_svg,
)


@dataclass
class BenchmarkCase:
    name: str
    data: bytes
    fmt: str
    category: str  # size: small, medium, large, vector
    content: str  # content type: photo, screenshot, graphic, etc.
    quality: int = 0  # source quality for lossy formats
    group: str = ""  # corpus group: high_res, standard, compact, deep_color


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


def build_all_cases() -> list[BenchmarkCase]:
    """Build the full benchmark suite."""
    cases = []
    cases.extend(_png_cases())
    cases.extend(_jpeg_cases())
    cases.extend(_webp_cases())
    cases.extend(_avif_cases())
    cases.extend(_heic_cases())
    cases.extend(_jxl_cases())
    cases.extend(_gif_cases())
    cases.extend(_bmp_cases())
    cases.extend(_tiff_cases())
    cases.extend(_svg_cases())
    return cases


def _lossy_photo_cases(fmt: str, encode_fmt: str, display: str) -> list[BenchmarkCase]:
    """Generate cases for a lossy photo format across all sizes and qualities."""
    cases = []
    for sname, w, h in CORE_SIZES:
        for q in LOSSY_QUALITIES:
            img = photo_like(w, h)
            data = encode_image(img, encode_fmt, quality=q)
            cases.append(
                BenchmarkCase(
                    name=f"{display} {sname} q={q} {w}x{h}",
                    data=data,
                    fmt=fmt,
                    category=sname,
                    content="photo",
                    quality=q,
                )
            )
    return cases


def _jpeg_cases() -> list[BenchmarkCase]:
    return _lossy_photo_cases("jpeg", "jpeg", "JPEG")


def _webp_cases() -> list[BenchmarkCase]:
    return _lossy_photo_cases("webp", "webp", "WebP")


def _avif_cases() -> list[BenchmarkCase]:
    """AVIF cases. Skipped if pillow-avif-plugin not available."""
    try:
        import pillow_avif  # noqa: F401

        return _lossy_photo_cases("avif", "avif", "AVIF")
    except ImportError:
        return []


def _heic_cases() -> list[BenchmarkCase]:
    """HEIC cases. Skipped if pillow-heif not available."""
    try:
        import pillow_heif  # noqa: F401

        return _lossy_photo_cases("heic", "heif", "HEIC")
    except ImportError:
        return []


def _jxl_cases() -> list[BenchmarkCase]:
    """JXL cases. Skipped if jxlpy/pillow-jxl not available."""
    try:
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            import jxlpy  # noqa: F401

        return _lossy_photo_cases("jxl", "jxl", "JXL")
    except ImportError:
        return []


# Content generators for PNG (3 BPP profiles: high, low, medium)
_PNG_CONTENT = [
    ("photo", photo_like),
    ("screenshot", screenshot_like),
    ("graphic", graphic_like),
]


def _png_cases() -> list[BenchmarkCase]:
    cases = []
    for sname, w, h in CORE_SIZES:
        for cname, gen in _PNG_CONTENT:
            img = gen(w, h)
            data = encode_image(img, "png")
            cases.append(
                BenchmarkCase(
                    name=f"PNG {sname} {cname} {w}x{h}",
                    data=data,
                    fmt="png",
                    category=sname,
                    content=cname,
                )
            )
    return cases


_GIF_CONTENT = [
    ("graphic", graphic_like),
    ("gradient", gradient),
]


def _gif_cases() -> list[BenchmarkCase]:
    cases = []
    for sname, w, h in CORE_SIZES:
        for cname, gen in _GIF_CONTENT:
            img = gen(w, h)
            data = encode_image(img, "gif")
            cases.append(
                BenchmarkCase(
                    name=f"GIF {sname} {cname} {w}x{h}",
                    data=data,
                    fmt="gif",
                    category=sname,
                    content=cname,
                )
            )
    return cases


_BMP_CONTENT = [
    ("screenshot", screenshot_like),
    ("photo", photo_like),
]


def _bmp_cases() -> list[BenchmarkCase]:
    cases = []
    for sname, w, h in CORE_SIZES:
        for cname, gen in _BMP_CONTENT:
            img = gen(w, h)
            data = encode_image(img, "bmp")
            cases.append(
                BenchmarkCase(
                    name=f"BMP {sname} {cname} {w}x{h}",
                    data=data,
                    fmt="bmp",
                    category=sname,
                    content=cname,
                )
            )
    return cases


_TIFF_COMPRESSIONS = [None, "tiff_lzw"]


def _tiff_cases() -> list[BenchmarkCase]:
    cases = []
    for sname, w, h in CORE_SIZES:
        for compression in _TIFF_COMPRESSIONS:
            comp_label = compression or "raw"
            img = photo_like(w, h)
            cases.append(
                BenchmarkCase(
                    name=f"TIFF {sname} {comp_label} {w}x{h}",
                    data=encode_image(img, "tiff", compression=compression),
                    fmt="tiff",
                    category=sname,
                    content="photo",
                )
            )
    return cases


def _svg_cases() -> list[BenchmarkCase]:
    cases = []
    for cname, gen in [("simple", svg_simple), ("complex", svg_complex), ("bloated", svg_bloated)]:
        data = gen()
        cases.append(
            BenchmarkCase(
                name=f"SVG {cname}",
                data=data,
                fmt="svg",
                category="vector",
                content=cname,
            )
        )
    for cname, gen in [("simple", svg_simple), ("complex", svg_complex), ("bloated", svg_bloated)]:
        svgz = svgz_from_svg(gen())
        cases.append(
            BenchmarkCase(
                name=f"SVGZ {cname}",
                data=svgz,
                fmt="svgz",
                category="vector",
                content=cname,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Corpus loader (real-world images from disk)
# ---------------------------------------------------------------------------

# Map file extensions to benchmark format strings
_EXT_TO_FMT = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".gif": "gif",
    ".bmp": "bmp",
    ".tiff": "tiff",
    ".tif": "tiff",
    ".avif": "avif",
    ".heic": "heic",
    ".heif": "heic",
    ".jxl": "jxl",
    ".svg": "svg",
    ".svgz": "svgz",
}

# Size thresholds (by largest dimension) for category labels
_SIZE_THRESHOLDS = [
    (200, "tiny"),
    (500, "small"),
    (1000, "medium"),
    (2000, "large"),
]


def _classify_size(data: bytes, fmt: str) -> str:
    """Classify image into a size category based on pixel dimensions."""
    if fmt in ("svg", "svgz"):
        return "vector"
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(data))
        max_dim = max(img.size)
        for threshold, label in _SIZE_THRESHOLDS:
            if max_dim <= threshold:
                return label
        return "xlarge"
    except Exception:
        # Fall back to file size heuristic
        size = len(data)
        if size < 50_000:
            return "small"
        if size < 500_000:
            return "medium"
        return "large"


def load_corpus_cases(corpus_dir: str | Path) -> list[BenchmarkCase]:
    """Load real-world images from a corpus directory as benchmark cases.

    Expected structure: corpus_dir/<category>/<name>.<ext>
    The parent folder name becomes the content category.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    cases = []
    for filepath in sorted(corpus_path.rglob("*")):
        if not filepath.is_file():
            continue
        ext = filepath.suffix.lower()
        fmt = _EXT_TO_FMT.get(ext)
        if fmt is None:
            continue  # skip non-image files (manifest.json, etc.)

        data = filepath.read_bytes()
        content = filepath.parent.name  # folder name = content category
        size_cat = _classify_size(data, fmt)
        name = f"{content}/{filepath.stem} [{fmt.upper()}]"

        cases.append(
            BenchmarkCase(
                name=name,
                data=data,
                fmt=fmt,
                category=size_cat,
                content=content,
            )
        )

    return cases

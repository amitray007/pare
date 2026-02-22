"""Benchmark case definitions.

Defines all image variants to test across sizes, content types, formats,
and quality levels. Each case is deterministic (seeded RNG).

Also supports loading real-world image files from a corpus directory
via load_corpus_cases().
"""

from dataclasses import dataclass
from pathlib import Path

from benchmarks.constants import (
    JPEG_LARGE_QUALITIES,
    JPEG_QUALITIES,
    JPEG_SCREENSHOT_QUALITY,
    JPEG_TINY_QUALITIES,
    LARGE_ALLOWED_CONTENT,
    SIZES,
    WEBP_LARGE_QUALITIES,
    WEBP_QUALITIES,
    sizes_excluding,
    sizes_matching,
)
from benchmarks.generators import (
    encode_image,
    gradient,
    graphic_like,
    palette_png,
    photo_like,
    screenshot_like,
    solid,
    svg_bloated,
    svg_complex,
    svg_simple,
    svgz_from_svg,
    transparent_png,
)


@dataclass
class BenchmarkCase:
    name: str
    data: bytes
    fmt: str
    category: str  # size: tiny, small-l, small-p, medium-l, medium-p, square, large-l, large-p
    content: str  # content type: photo, screenshot, graphic, etc.
    quality: int = 0  # source quality for JPEG/WebP


# ---------------------------------------------------------------------------
# Content generators
# ---------------------------------------------------------------------------

PNG_GENERATORS = [
    ("photo", photo_like),
    ("screenshot", screenshot_like),
    ("graphic", graphic_like),
    ("gradient", gradient),
    ("solid", solid),
    ("transparent", transparent_png),
]

SVG_GENERATORS = [
    ("simple", svg_simple),
    ("complex", svg_complex),
    ("bloated", svg_bloated),
]


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


def build_all_cases() -> list[BenchmarkCase]:
    """Build the full benchmark suite."""
    cases = []
    cases.extend(_png_cases())
    cases.extend(_jpeg_cases())
    cases.extend(_webp_cases())
    cases.extend(_gif_cases())
    cases.extend(_svg_cases())
    cases.extend(_other_cases())
    cases.extend(_avif_heic_cases())
    cases.extend(_jxl_cases())
    return cases


def _png_cases() -> list[BenchmarkCase]:
    cases = []
    for sname, w, h in SIZES:
        is_large = sname.startswith("large")
        for cname, gen in PNG_GENERATORS:
            if is_large and cname not in LARGE_ALLOWED_CONTENT:
                continue
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
    # Palette PNG — small and medium landscape only
    for sname, w, h in sizes_matching("small-l", "medium-l"):
        img = palette_png(w, h)
        data = encode_image(img, "png")
        cases.append(
            BenchmarkCase(
                name=f"PNG {sname} palette {w}x{h}",
                data=data,
                fmt="png",
                category=sname,
                content="palette",
            )
        )
    return cases


def _jpeg_cases() -> list[BenchmarkCase]:
    cases = []
    for sname, w, h in SIZES:
        is_large = sname.startswith("large")
        is_tiny = sname == "tiny"
        for q in JPEG_QUALITIES:
            if is_large and q not in JPEG_LARGE_QUALITIES:
                continue
            if is_tiny and q not in JPEG_TINY_QUALITIES:
                continue
            img = photo_like(w, h)
            data = encode_image(img, "jpeg", quality=q)
            cases.append(
                BenchmarkCase(
                    name=f"JPEG {sname} q={q} {w}x{h}",
                    data=data,
                    fmt="jpeg",
                    category=sname,
                    content="photo",
                    quality=q,
                )
            )
    # Screenshot as JPEG — medium and large landscape
    for sname, w, h in sizes_matching("medium-l", "large-l"):
        img = screenshot_like(w, h)
        data = encode_image(img, "jpeg", quality=JPEG_SCREENSHOT_QUALITY)
        cases.append(
            BenchmarkCase(
                name=f"JPEG {sname} screenshot q={JPEG_SCREENSHOT_QUALITY} {w}x{h}",
                data=data,
                fmt="jpeg",
                category=sname,
                content="screenshot",
                quality=JPEG_SCREENSHOT_QUALITY,
            )
        )
    return cases


def _webp_cases() -> list[BenchmarkCase]:
    cases = []
    for sname, w, h in sizes_excluding("tiny", "small-p", "medium-p", "large-p"):
        is_large = sname.startswith("large")
        for q in WEBP_QUALITIES:
            if is_large and q not in WEBP_LARGE_QUALITIES:
                continue
            img = photo_like(w, h)
            data = encode_image(img, "webp", quality=q)
            cases.append(
                BenchmarkCase(
                    name=f"WebP {sname} q={q} {w}x{h}",
                    data=data,
                    fmt="webp",
                    category=sname,
                    content="photo",
                    quality=q,
                )
            )
    return cases


def _gif_cases() -> list[BenchmarkCase]:
    cases = []
    for sname, w, h in sizes_excluding("large", "small-p", "medium-p"):
        for cname, gen in [("graphic", graphic_like), ("gradient", gradient)]:
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


def _svg_cases() -> list[BenchmarkCase]:
    cases = []
    for cname, gen in SVG_GENERATORS:
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
    for cname, gen in SVG_GENERATORS:
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


TIFF_COMPRESSIONS = [None, "tiff_lzw"]  # None = raw/uncompressed


def _other_cases() -> list[BenchmarkCase]:
    """BMP and TIFF benchmark cases."""
    sname, w, h = sizes_matching("small-l")[0]
    cases = []

    # BMP cases — RGB and RGBA (graphic_like produces RGBA)
    img = screenshot_like(w, h)
    cases.append(
        BenchmarkCase(
            name=f"BMP screenshot {w}x{h}",
            data=encode_image(img, "bmp"),
            fmt="bmp",
            category=sname,
            content="screenshot",
        )
    )
    img = photo_like(w, h)
    cases.append(
        BenchmarkCase(
            name=f"BMP photo {w}x{h}",
            data=encode_image(img, "bmp"),
            fmt="bmp",
            category=sname,
            content="photo",
        )
    )
    img = graphic_like(w, h)
    cases.append(
        BenchmarkCase(
            name=f"BMP graphic RGBA {w}x{h}",
            data=encode_image(img, "bmp"),
            fmt="bmp",
            category=sname,
            content="graphic",
        )
    )

    # TIFF cases — each content type × source compression level
    tiff_content = [
        ("photo", photo_like),
        ("screenshot", screenshot_like),
        ("graphic", graphic_like),
    ]
    for compression in TIFF_COMPRESSIONS:
        comp_label = compression or "raw"
        for cname, gen in tiff_content:
            img = gen(w, h)
            cases.append(
                BenchmarkCase(
                    name=f"TIFF {cname} {comp_label} {w}x{h}",
                    data=encode_image(img, "tiff", compression=compression),
                    fmt="tiff",
                    category=sname,
                    content=cname,
                )
            )

    return cases


AVIF_QUALITIES = [95, 75, 50]
HEIC_QUALITIES = [95, 75, 50]


def _avif_heic_cases() -> list[BenchmarkCase]:
    """AVIF and HEIC benchmark cases. Skipped if dependencies not available."""
    sname, w, h = sizes_matching("small-l")[0]
    cases = []

    # AVIF cases — require pillow-avif-plugin
    try:
        import pillow_avif  # noqa: F401

        for q in AVIF_QUALITIES:
            img = photo_like(w, h)
            cases.append(
                BenchmarkCase(
                    name=f"AVIF photo q={q} {w}x{h}",
                    data=encode_image(img, "avif", quality=q),
                    fmt="avif",
                    category=sname,
                    content="photo",
                    quality=q,
                )
            )
    except ImportError:
        pass

    # HEIC cases — require pillow-heif
    try:
        import pillow_heif  # noqa: F401

        for q in HEIC_QUALITIES:
            img = photo_like(w, h)
            cases.append(
                BenchmarkCase(
                    name=f"HEIC photo q={q} {w}x{h}",
                    data=encode_image(img, "heif", quality=q),
                    fmt="heic",
                    category=sname,
                    content="photo",
                    quality=q,
                )
            )
    except ImportError:
        pass

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


JXL_QUALITIES = [95, 75, 50]


def _jxl_cases() -> list[BenchmarkCase]:
    """JPEG XL benchmark cases. Skipped if jxlpy not available."""
    sname, w, h = sizes_matching("small-l")[0]
    cases = []

    try:
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            import jxlpy  # noqa: F401

        for q in JXL_QUALITIES:
            img = photo_like(w, h)
            cases.append(
                BenchmarkCase(
                    name=f"JXL photo q={q} {w}x{h}",
                    data=encode_image(img, "jxl", quality=q),
                    fmt="jxl",
                    category=sname,
                    content="photo",
                    quality=q,
                )
            )
    except ImportError:
        pass

    return cases

"""Convert the Unsplash JPEG corpus into multiple formats.

Usage:
    python scripts/convert_corpus_formats.py

Converts each JPEG in tests/corpus/ into PNG, WebP, BMP, TIFF, GIF,
and optionally AVIF, HEIC, and JXL (if libraries are available).
Skips SVG/SVGZ/APNG as they don't apply to photographic content.

The original JPEGs are kept. New files are placed alongside with
the same name but different extension.
"""

import json
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "corpus"

# Formats to convert to, with Pillow save kwargs
# Each entry: (extension, pillow_format, save_kwargs, requires_module)
CONVERSIONS = [
    ("png", "PNG", {}, None),
    ("webp", "WebP", {"quality": 90}, None),
    ("bmp", "BMP", {}, None),
    ("tiff", "TIFF", {}, None),
    ("gif", "GIF", {}, None),
]

# Optional formats that need extra libraries
OPTIONAL_CONVERSIONS = [
    ("avif", "AVIF", {"quality": 80}, "pillow_avif"),
    ("heic", "HEIF", {"quality": 80}, "pillow_heif"),
    ("jxl", "JXL", {"quality": 80}, "jxlpy"),
]


def check_optional_formats():
    """Check which optional format libraries are available."""
    available = []
    for ext, fmt, kwargs, module in OPTIONAL_CONVERSIONS:
        try:
            if module == "pillow_avif":
                import pillow_avif  # noqa: F401
            elif module == "pillow_heif":
                import pillow_heif
                pillow_heif.register_heif_opener()
            elif module == "jxlpy":
                try:
                    import pillow_jxl  # noqa: F401
                except ImportError:
                    import jxlpy  # noqa: F401
            available.append((ext, fmt, kwargs, module))
            print(f"  [OK] {ext.upper()} support available ({module})")
        except ImportError:
            print(f"  [--] {ext.upper()} not available (missing {module})")
    return available


def convert_image(src: Path, ext: str, fmt: str, save_kwargs: dict) -> Path | None:
    """Convert a JPEG to another format. Returns output path or None on failure."""
    dest = src.with_suffix(f".{ext}")
    if dest.exists():
        return dest

    try:
        img = Image.open(src)

        # GIF and BMP don't support RGBA well from JPEG, but JPEG is always RGB
        # For GIF, quantize to palette
        if fmt == "GIF":
            img = img.convert("RGB").quantize(colors=256)
        elif fmt == "BMP":
            img = img.convert("RGB")

        img.save(str(dest), format=fmt, **save_kwargs)
        return dest
    except Exception as e:
        print(f"    FAILED {ext}: {e}")
        return None


def main():
    print("Checking available format libraries...")
    optional = check_optional_formats()
    all_conversions = CONVERSIONS + optional
    print(f"\nWill convert to: {', '.join(ext.upper() for ext, *_ in all_conversions)}")

    # Find all JPEGs in corpus
    jpegs = sorted(CORPUS_DIR.rglob("*.jpg"))
    if not jpegs:
        print(f"\nNo JPEGs found in {CORPUS_DIR}. Run download_unsplash_corpus.py first.")
        sys.exit(1)

    print(f"\nFound {len(jpegs)} JPEG source files")
    print(f"Converting to {len(all_conversions)} formats = ~{len(jpegs) * len(all_conversions)} new files\n")

    stats = {ext: {"created": 0, "skipped": 0, "failed": 0} for ext, *_ in all_conversions}
    current_category = None

    for jpeg in jpegs:
        category = jpeg.parent.name
        if category != current_category:
            current_category = category
            print(f"\n{'='*50}")
            print(f"  {category}")
            print(f"{'='*50}")

        print(f"  {jpeg.stem}:")
        for ext, fmt, kwargs, _ in all_conversions:
            dest = jpeg.with_suffix(f".{ext}")
            if dest.exists():
                stats[ext]["skipped"] += 1
                size_kb = dest.stat().st_size / 1024
                print(f"    {ext:5s}: exists ({size_kb:.0f} KB)")
            else:
                result = convert_image(jpeg, ext, fmt, kwargs)
                if result:
                    stats[ext]["created"] += 1
                    size_kb = result.stat().st_size / 1024
                    print(f"    {ext:5s}: created ({size_kb:.0f} KB)")
                else:
                    stats[ext]["failed"] += 1

    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary:")
    print(f"{'='*60}")
    print(f"  {'Format':<8} {'Created':>8} {'Skipped':>8} {'Failed':>8}")
    print(f"  {'-'*36}")
    for ext, *_ in all_conversions:
        s = stats[ext]
        print(f"  {ext.upper():<8} {s['created']:>8} {s['skipped']:>8} {s['failed']:>8}")

    # Show total corpus size
    total_bytes = sum(f.stat().st_size for f in CORPUS_DIR.rglob("*") if f.is_file() and f.name != "manifest.json")
    total_files = sum(1 for f in CORPUS_DIR.rglob("*") if f.is_file() and f.name != "manifest.json")
    print(f"\n  Total corpus: {total_files} files, {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Location: {CORPUS_DIR}")


if __name__ == "__main__":
    main()

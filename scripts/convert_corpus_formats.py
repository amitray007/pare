"""Convert the Unsplash corpus into additional formats.

Usage:
    python scripts/convert_corpus_formats.py

Converts source images in tests/corpus/ into formats not available
from the CDN.  The download script already provides:
  - JPEG (fm=jpg)  — from Unsplash CDN
  - AVIF (fm=avif) — from Unsplash CDN
  - PNG  (fm=png)  — from Unsplash CDN

This script converts those sources into:
  - WebP, BMP, TIFF, GIF  — from JPEG source
  - HEIC                    — from PNG source (lossless, avoids JPEG artifacts)
  - JXL                     — from PNG source (lossless, avoids JPEG artifacts)

Skips SVG/SVGZ/APNG as they don't apply to photographic content.
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "corpus"

# Formats converted from JPEG source (lossy-to-lossy or lossy-to-lossless is fine)
# Each entry: (extension, pillow_format, save_kwargs)
JPEG_CONVERSIONS = [
    ("webp", "WebP", {"quality": 90}),
    ("bmp", "BMP", {}),
    ("tiff", "TIFF", {}),
    ("gif", "GIF", {}),
]

# Formats converted from PNG source (lossless source avoids double-compression artifacts)
# Each entry: (extension, pillow_format, save_kwargs, requires_module)
PNG_CONVERSIONS = [
    ("heic", "HEIF", {"quality": 80}, "pillow_heif"),
    ("jxl", "JXL", {"quality": 80}, "jxlpy"),
]


def check_optional_formats():
    """Check which optional format libraries are available."""
    available = []
    for ext, fmt, kwargs, module in PNG_CONVERSIONS:
        try:
            if module == "pillow_heif":
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
    """Convert an image to another format. Returns output path or None on failure."""
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
        elif fmt in ("HEIF", "JXL", "WebP"):
            # Ensure RGB mode for lossy formats
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

        img.save(str(dest), format=fmt, **save_kwargs)
        return dest
    except Exception as e:
        print(f"    FAILED {ext}: {e}")
        return None


def main():
    print("Checking available format libraries...")
    optional = check_optional_formats()

    print(f"\nFrom JPEG: {', '.join(ext.upper() for ext, *_ in JPEG_CONVERSIONS)}")
    if optional:
        print(f"From PNG:  {', '.join(ext.upper() for ext, *_ in optional)}")

    # Find all JPEGs in corpus (source for WebP/BMP/TIFF/GIF)
    jpegs = sorted(CORPUS_DIR.rglob("*.jpg"))
    if not jpegs:
        print(f"\nNo JPEGs found in {CORPUS_DIR}. Run download_unsplash_corpus.py first.")
        sys.exit(1)

    # Find all PNGs in corpus (source for HEIC/JXL — lossless avoids double compression)
    pngs = sorted(CORPUS_DIR.rglob("*.png"))

    print(f"\nFound {len(jpegs)} JPEG source files")
    print(f"Found {len(pngs)} PNG source files")

    total_formats = len(JPEG_CONVERSIONS) + len(optional)
    print(f"Converting to {total_formats} formats\n")

    all_exts = [ext for ext, *_ in JPEG_CONVERSIONS] + [ext for ext, *_ in optional]
    stats = {ext: {"created": 0, "skipped": 0, "failed": 0} for ext in all_exts}
    current_category = None

    # Convert from JPEG sources (WebP, BMP, TIFF, GIF)
    for jpeg in jpegs:
        category = jpeg.parent.name
        if category != current_category:
            current_category = category
            print(f"\n{'='*50}")
            print(f"  {category}")
            print(f"{'='*50}")

        print(f"  {jpeg.stem}:")
        for ext, fmt, kwargs in JPEG_CONVERSIONS:
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

    # Convert from PNG sources (HEIC, JXL — lossless source for best quality)
    if optional and pngs:
        current_category = None
        print(f"\n\n{'='*60}")
        print("  Converting from PNG source (lossless) -> HEIC, JXL")
        print(f"{'='*60}")

        for png in pngs:
            category = png.parent.name
            if category != current_category:
                current_category = category
                print(f"\n{'='*50}")
                print(f"  {category}")
                print(f"{'='*50}")

            print(f"  {png.stem}:")
            for ext, fmt, kwargs, _ in optional:
                dest = png.with_suffix(f".{ext}")
                if dest.exists():
                    stats[ext]["skipped"] += 1
                    size_kb = dest.stat().st_size / 1024
                    print(f"    {ext:5s}: exists ({size_kb:.0f} KB)")
                else:
                    result = convert_image(png, ext, fmt, kwargs)
                    if result:
                        stats[ext]["created"] += 1
                        size_kb = result.stat().st_size / 1024
                        print(f"    {ext:5s}: created ({size_kb:.0f} KB)")
                    else:
                        stats[ext]["failed"] += 1
    elif optional and not pngs:
        print("\n  WARNING: No PNG files found. Run download script with --formats png first.")
        print("  HEIC/JXL conversion skipped (needs lossless PNG source).")

    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary:")
    print(f"{'='*60}")
    print(f"  {'Format':<8} {'Source':<8} {'Created':>8} {'Skipped':>8} {'Failed':>8}")
    print(f"  {'-'*44}")
    for ext, *_ in JPEG_CONVERSIONS:
        s = stats[ext]
        print(f"  {ext.upper():<8} {'JPEG':<8} {s['created']:>8} {s['skipped']:>8} {s['failed']:>8}")
    for ext, *_ in optional:
        s = stats[ext]
        print(f"  {ext.upper():<8} {'PNG':<8} {s['created']:>8} {s['skipped']:>8} {s['failed']:>8}")

    # Show total corpus size
    total_bytes = sum(
        f.stat().st_size for f in CORPUS_DIR.rglob("*") if f.is_file() and f.name != "manifest.json"
    )
    total_files = sum(1 for f in CORPUS_DIR.rglob("*") if f.is_file() and f.name != "manifest.json")
    print(f"\n  Total corpus: {total_files} files, {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Location: {CORPUS_DIR}")


if __name__ == "__main__":
    main()

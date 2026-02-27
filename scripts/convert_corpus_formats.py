"""Convert corpus images into additional formats.

Usage:
    python scripts/convert_corpus_formats.py
    python scripts/convert_corpus_formats.py --dry-run

Converts source images in tests/corpus/{high_res,standard,compact}/ into
formats not available from the CDN.  The download script already provides:
  - JPEG (fm=jpg)  -- from Unsplash CDN
  - AVIF (fm=avif) -- from Unsplash CDN
  - PNG  (fm=png)  -- from Unsplash CDN
  - WebP (fm=webp) -- from Unsplash CDN

This script converts those sources into:
  - BMP, TIFF, GIF  -- from JPEG source
  - HEIC             -- from PNG source (lossless, avoids JPEG artifacts)
  - JXL              -- from PNG source (lossless, avoids JPEG artifacts)

Skips deep_color/ directory (those are all native samples).
Updates groups.json with converted files.
"""

import json
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "corpus"
GROUPS_JSON = CORPUS_DIR / "groups.json"

# Group directories to convert (skip deep_color — native samples only)
CONVERT_GROUPS = ["high_res", "standard", "compact"]

# Formats converted from JPEG source
# Each entry: (extension, pillow_format, save_kwargs)
JPEG_CONVERSIONS = [
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

        if fmt == "GIF":
            img = img.convert("RGB").quantize(colors=256)
        elif fmt == "BMP":
            img = img.convert("RGB")
        elif fmt in ("HEIF", "JXL", "WebP"):
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

        img.save(str(dest), format=fmt, **save_kwargs)
        return dest
    except Exception as e:
        print(f"    FAILED {ext}: {e}")
        return None


def _fmt_from_ext(ext: str) -> str:
    """Map file extension to format name."""
    return {
        "bmp": "bmp",
        "tiff": "tiff",
        "gif": "gif",
        "heic": "heic",
        "jxl": "jxl",
    }.get(ext, ext)


def main():
    dry_run = "--dry-run" in sys.argv

    print("Checking available format libraries...")
    optional = check_optional_formats()

    print(f"\nFrom JPEG: {', '.join(ext.upper() for ext, *_ in JPEG_CONVERSIONS)}")
    if optional:
        print(f"From PNG:  {', '.join(ext.upper() for ext, *_ in optional)}")

    all_exts = [ext for ext, *_ in JPEG_CONVERSIONS] + [ext for ext, *_ in optional]
    stats = {ext: {"created": 0, "skipped": 0, "failed": 0} for ext in all_exts}
    new_files: dict[str, list[dict]] = {}  # group -> [file_info]

    for group in CONVERT_GROUPS:
        group_dir = CORPUS_DIR / group
        if not group_dir.is_dir():
            print(f"\n  Skipping {group}/ (not found)")
            continue

        jpegs = sorted(group_dir.glob("*.jpg"))
        pngs = sorted(group_dir.glob("*.png"))

        if not jpegs and not pngs:
            print(f"\n  Skipping {group}/ (no source files)")
            continue

        print(f"\n{'='*50}")
        print(f"  {group} ({len(jpegs)} JPEG, {len(pngs)} PNG)")
        print(f"{'='*50}")

        # Convert from JPEG sources (BMP, TIFF, GIF)
        for jpeg in jpegs:
            print(f"  {jpeg.stem}:")
            for ext, fmt, kwargs in JPEG_CONVERSIONS:
                dest = jpeg.with_suffix(f".{ext}")
                if dest.exists():
                    stats[ext]["skipped"] += 1
                    size_kb = dest.stat().st_size / 1024
                    print(f"    {ext:5s}: exists ({size_kb:.0f} KB)")
                    if not dry_run:
                        new_files.setdefault(group, []).append(
                            {
                                "path": f"{group}/{dest.name}",
                                "format": _fmt_from_ext(ext),
                                "source_type": "lossless",
                                "category": "medium",
                                "size_bytes": dest.stat().st_size,
                            }
                        )
                elif dry_run:
                    print(f"    {ext:5s}: would create")
                else:
                    result = convert_image(jpeg, ext, fmt, kwargs)
                    if result:
                        stats[ext]["created"] += 1
                        size_kb = result.stat().st_size / 1024
                        print(f"    {ext:5s}: created ({size_kb:.0f} KB)")
                        new_files.setdefault(group, []).append(
                            {
                                "path": f"{group}/{result.name}",
                                "format": _fmt_from_ext(ext),
                                "source_type": "lossless",
                                "category": "medium",
                                "size_bytes": result.stat().st_size,
                            }
                        )
                    else:
                        stats[ext]["failed"] += 1

        # Convert from PNG sources (HEIC, JXL)
        if optional and pngs:
            for png in pngs:
                print(f"  {png.stem}:")
                for ext, fmt, kwargs, _ in optional:
                    dest = png.with_suffix(f".{ext}")
                    if dest.exists():
                        stats[ext]["skipped"] += 1
                        size_kb = dest.stat().st_size / 1024
                        print(f"    {ext:5s}: exists ({size_kb:.0f} KB)")
                        if not dry_run:
                            new_files.setdefault(group, []).append(
                                {
                                    "path": f"{group}/{dest.name}",
                                    "format": _fmt_from_ext(ext),
                                    "source_type": "lossless",
                                    "category": "medium",
                                    "size_bytes": dest.stat().st_size,
                                }
                            )
                    elif dry_run:
                        print(f"    {ext:5s}: would create")
                    else:
                        result = convert_image(png, ext, fmt, kwargs)
                        if result:
                            stats[ext]["created"] += 1
                            size_kb = result.stat().st_size / 1024
                            print(f"    {ext:5s}: created ({size_kb:.0f} KB)")
                            new_files.setdefault(group, []).append(
                                {
                                    "path": f"{group}/{result.name}",
                                    "format": _fmt_from_ext(ext),
                                    "source_type": "lossless",
                                    "category": "medium",
                                    "size_bytes": result.stat().st_size,
                                }
                            )
                        else:
                            stats[ext]["failed"] += 1

    # Update groups.json with converted files
    if not dry_run and new_files and GROUPS_JSON.exists():
        manifest = json.loads(GROUPS_JSON.read_text(encoding="utf-8"))
        for group_key, files in new_files.items():
            if group_key in manifest.get("groups", {}):
                existing_paths = {f["path"] for f in manifest["groups"][group_key].get("files", [])}
                for f in files:
                    if f["path"] not in existing_paths:
                        manifest["groups"][group_key].setdefault("files", []).append(f)
        GROUPS_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nUpdated {GROUPS_JSON}")

    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary:")
    print(f"{'='*60}")
    print(f"  {'Format':<8} {'Source':<8} {'Created':>8} {'Skipped':>8} {'Failed':>8}")
    print(f"  {'-'*44}")
    for ext, *_ in JPEG_CONVERSIONS:
        s = stats[ext]
        print(
            f"  {ext.upper():<8} {'JPEG':<8} {s['created']:>8} {s['skipped']:>8} {s['failed']:>8}"
        )
    for ext, *_ in optional:
        s = stats[ext]
        print(f"  {ext.upper():<8} {'PNG':<8} {s['created']:>8} {s['skipped']:>8} {s['failed']:>8}")

    # Show total corpus size
    total_bytes = sum(
        f.stat().st_size
        for f in CORPUS_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() not in (".json",)
    )
    total_files = sum(
        1 for f in CORPUS_DIR.rglob("*") if f.is_file() and f.suffix.lower() not in (".json",)
    )
    print(f"\n  Total corpus: {total_files} files, {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Location: {CORPUS_DIR}")


if __name__ == "__main__":
    main()

"""Download a curated test corpus from Unsplash.

Usage:
    python scripts/download_unsplash_corpus.py

Requires UNSPLASH_ACCESS_KEY env var or pass --key.
Downloads real-world images across categories and sizes
into tests/corpus/ for compression and estimation testing.

Downloads each image in multiple formats via the Unsplash/Imgix CDN:
  - JPEG (fm=jpg, q=90)  — primary lossy format
  - AVIF (fm=avif, q=80) — modern lossy, CDN-native encoding
  - PNG  (fm=png)         — lossless from original source

This gives us CDN-native AVIF and PNG files encoded from the original
source rather than re-transcoded from downloaded JPEGs.  HEIC is not
supported by the CDN, so it is converted from the PNG source by the
convert_corpus_formats.py script.

Additionally downloads native AVIF test images from the link-u/avif-sample-images
repository for diverse AVIF encoding profiles (8/10/12-bit, YUV420/422/444, alpha).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Categories and search queries
# ---------------------------------------------------------------------------

# Each entry: (folder_name, search_query, count)
# We pick diverse queries to cover different compression characteristics.
CATEGORIES = [
    ("landscape", "landscape nature", 5),
    ("portrait", "portrait face person", 4),
    ("architecture", "architecture building", 4),
    ("texture", "texture pattern close-up", 4),
    ("macro", "macro detail", 3),
    ("lowlight", "night dark moody", 3),
    ("highcontrast", "high contrast black white", 3),
    ("colorful", "colorful vibrant bright", 3),
    ("monochrome", "black and white monochrome", 3),
    ("text_heavy", "sign typography poster", 3),
    ("aerial", "aerial drone top-down", 3),
    ("food", "food cooking dish", 3),
    ("abstract", "abstract minimalist", 3),
]

# Sizes to download each image at (label, width)
SIZES = [
    ("small", 400),
    ("medium", 1200),
    ("large", 2400),
]

# Formats to download from the Unsplash/Imgix CDN.
# Each entry: (extension, CDN fm= param, extra URL params)
CDN_FORMATS = [
    ("jpg", "jpg", "q=90"),
    ("avif", "avif", "q=80"),
    ("png", "png", ""),
]

# Native AVIF test images from link-u/avif-sample-images (CC-BY-SA 4.0).
# Curated selection covering diverse encoding profiles.
_LINKU_BASE = "https://raw.githubusercontent.com/link-u/avif-sample-images/master"
LINKU_AVIF_SAMPLES = [
    # Hato (pigeon, 3082x2048) — large photo
    ("hato_8bit_yuv420", f"{_LINKU_BASE}/hato.profile0.8bpc.yuv420.avif"),
    ("hato_10bit_yuv422", f"{_LINKU_BASE}/hato.profile2.10bpc.yuv422.avif"),
    ("hato_12bit_yuv422", f"{_LINKU_BASE}/hato.profile2.12bpc.yuv422.avif"),
    # Fox parade (1204x800) — medium photo
    ("fox_8bit_yuv420", f"{_LINKU_BASE}/fox.profile0.8bpc.yuv420.avif"),
    ("fox_10bit_yuv444", f"{_LINKU_BASE}/fox.profile1.10bpc.yuv444.avif"),
    ("fox_12bit_yuv422", f"{_LINKU_BASE}/fox.profile2.12bpc.yuv422.avif"),
    # Kimono (722x1024) — small portrait
    ("kimono_standard", f"{_LINKU_BASE}/kimono.avif"),
    # Fox — 12-bit YUV420 (different chroma subsampling from yuv422/yuv444 above)
    ("fox_12bit_yuv420", f"{_LINKU_BASE}/fox.profile2.12bpc.yuv420.avif"),
]

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "corpus"
MANIFEST_FILE = CORPUS_DIR / "manifest.json"


def fetch_json(url: str, access_key: str) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Client-ID {access_key}",
            "Accept-Version": "v1",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file, return True on success."""
    if dest.exists() and not force:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  FAILED: {e}")
        return False


def search_photos(query: str, count: int, access_key: str, page: int = 1) -> list[dict]:
    """Search Unsplash for photos matching a query."""
    url = (
        f"https://api.unsplash.com/search/photos"
        f"?query={urllib.parse.quote(query)}"
        f"&per_page={count}"
        f"&page={page}"
        f"&orientation=landscape"
        f"&order_by=relevant"
    )
    data = fetch_json(url, access_key)
    return data.get("results", [])


def build_download_url(photo: dict, width: int, fmt: str = "jpg", extra: str = "q=90") -> str:
    """Build a sized download URL from an Unsplash photo object."""
    raw = photo["urls"]["raw"]
    sep = "&" if "?" in raw else "?"
    params = f"w={width}&fm={fmt}&fit=crop"
    if extra:
        params += f"&{extra}"
    return f"{raw}{sep}{params}"


def download_linku_avif_samples(dry_run: bool = False, force: bool = False) -> tuple[int, int, int]:
    """Download native AVIF samples from link-u/avif-sample-images."""
    dest_dir = CORPUS_DIR / "avif_native"
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded, skipped, failed = 0, 0, 0

    print(f"\n{'='*60}")
    print("Native AVIF samples (link-u/avif-sample-images)")
    print(f"{'='*60}")

    for name, url in LINKU_AVIF_SAMPLES:
        dest = dest_dir / f"{name}.avif"

        if dry_run:
            print(f"  Would download: {name} -> {dest.relative_to(CORPUS_DIR)}")
            continue

        if dest.exists() and not force:
            size_kb = dest.stat().st_size / 1024
            print(f"  {name}: exists ({size_kb:.0f} KB)")
            skipped += 1
        else:
            ok = download_file(url, dest, force=force)
            if ok:
                size_kb = dest.stat().st_size / 1024
                print(f"  {name}: downloaded ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                failed += 1
            time.sleep(0.3)

    return downloaded, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Download Unsplash test corpus")
    parser.add_argument(
        "--key",
        default=os.environ.get("UNSPLASH_ACCESS_KEY"),
        help="Unsplash API access key (env: UNSPLASH_ACCESS_KEY)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be downloaded without downloading"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["small", "medium", "large"],
        default=None,
        help="Only download specific sizes (default: all)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["jpg", "avif", "png"],
        default=None,
        help="Only download specific CDN formats (default: all)",
    )
    parser.add_argument(
        "--skip-linku", action="store_true", help="Skip downloading link-u AVIF samples"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist (replaces old converted versions with CDN-native)",
    )
    args = parser.parse_args()

    if not args.key:
        print("ERROR: Set UNSPLASH_ACCESS_KEY env var or pass --key")
        sys.exit(1)

    access_key = args.key
    sizes = [(label, w) for label, w in SIZES if args.sizes is None or label in args.sizes]
    cdn_formats = [
        (ext, fm, extra)
        for ext, fm, extra in CDN_FORMATS
        if args.formats is None or ext in args.formats
    ]

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for category, query, count in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Category: {category} (query: '{query}', {count} photos)")
        print(f"{'='*60}")

        photos = search_photos(query, count, access_key)
        if not photos:
            print(f"  No results for '{query}', skipping")
            continue

        # Rate limiting: Unsplash allows 50 req/hour for demo apps
        time.sleep(1)

        for i, photo in enumerate(photos[:count]):
            photo_id = photo["id"]
            author = photo["user"]["name"]
            description = (
                photo.get("alt_description") or photo.get("description") or "no description"
            )
            orig_w = photo["width"]
            orig_h = photo["height"]

            print(f"\n  [{i+1}/{count}] {photo_id} by {author}")
            print(f"    {description[:80]}")
            print(f"    Original: {orig_w}x{orig_h}")

            manifest_key = f"{category}/{photo_id}"
            manifest[manifest_key] = {
                "id": photo_id,
                "category": category,
                "author": author,
                "description": description[:200],
                "original_size": f"{orig_w}x{orig_h}",
                "unsplash_url": photo["links"]["html"],
                "files": {},
            }

            for size_label, target_width in sizes:
                # Skip sizes larger than the original
                if target_width > orig_w:
                    effective_width = orig_w
                else:
                    effective_width = target_width

                for ext, fm, extra in cdn_formats:
                    filename = f"{photo_id}_{size_label}.{ext}"
                    dest = CORPUS_DIR / category / filename
                    url = build_download_url(photo, effective_width, fmt=fm, extra=extra)

                    if args.dry_run:
                        print(
                            f"    Would download: {size_label} {ext.upper()} ({effective_width}px)"
                            f" -> {dest.relative_to(CORPUS_DIR)}"
                        )
                        continue

                    if dest.exists() and not args.force:
                        size_kb = dest.stat().st_size / 1024
                        print(
                            f"    {size_label} {ext.upper()} ({effective_width}px):"
                            f" exists ({size_kb:.0f} KB)"
                        )
                        total_skipped += 1
                    else:
                        ok = download_file(url, dest, force=args.force)
                        if ok:
                            size_kb = dest.stat().st_size / 1024
                            action = "replaced" if args.force and dest.exists() else "downloaded"
                            print(
                                f"    {size_label} {ext.upper()} ({effective_width}px):"
                                f" {action} ({size_kb:.0f} KB)"
                            )
                            total_downloaded += 1
                        else:
                            total_failed += 1
                        # Be nice to the API
                        time.sleep(0.5)

                    file_key = f"{size_label}_{ext}" if ext != "jpg" else size_label
                    if dest.exists():
                        manifest[manifest_key]["files"][file_key] = {
                            "path": str(dest.relative_to(CORPUS_DIR)),
                            "size_bytes": dest.stat().st_size,
                        }

    # Download native AVIF samples from link-u
    if not args.skip_linku:
        d, s, f = download_linku_avif_samples(dry_run=args.dry_run, force=args.force)
        total_downloaded += d
        total_skipped += s
        total_failed += f

    # Save manifest
    if not args.dry_run:
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved to {MANIFEST_FILE}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Skipped (exists): {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Total categories: {len(CATEGORIES)}")
    print(f"  CDN formats: {', '.join(ext.upper() for ext, *_ in cdn_formats)}")
    print(f"  Corpus directory: {CORPUS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

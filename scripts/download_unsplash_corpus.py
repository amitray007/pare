"""Download a curated test corpus from Unsplash.

Usage:
    python scripts/download_unsplash_corpus.py

Requires UNSPLASH_ACCESS_KEY env var or pass --key.
Downloads ~40-50 real-world images across categories and sizes
into tests/corpus/ for compression and estimation testing.
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


def download_file(url: str, dest: Path) -> bool:
    """Download a file, return True on success."""
    if dest.exists():
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


def build_download_url(photo: dict, width: int) -> str:
    """Build a sized download URL from an Unsplash photo object."""
    raw = photo["urls"]["raw"]
    # raw URL is like: https://images.unsplash.com/photo-xxx?ixid=...&ixlib=...
    # We append width and quality params
    sep = "&" if "?" in raw else "?"
    return f"{raw}{sep}w={width}&q=90&fm=jpg&fit=crop"


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
    args = parser.parse_args()

    if not args.key:
        print("ERROR: Set UNSPLASH_ACCESS_KEY env var or pass --key")
        sys.exit(1)

    access_key = args.key
    sizes = [(label, w) for label, w in SIZES if args.sizes is None or label in args.sizes]

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

                filename = f"{photo_id}_{size_label}.jpg"
                dest = CORPUS_DIR / category / filename
                url = build_download_url(photo, effective_width)

                if args.dry_run:
                    print(
                        f"    Would download: {size_label} ({effective_width}px) -> {dest.relative_to(CORPUS_DIR)}"
                    )
                    continue

                if dest.exists():
                    size_kb = dest.stat().st_size / 1024
                    print(f"    {size_label} ({effective_width}px): exists ({size_kb:.0f} KB)")
                    total_skipped += 1
                else:
                    ok = download_file(url, dest)
                    if ok:
                        size_kb = dest.stat().st_size / 1024
                        print(
                            f"    {size_label} ({effective_width}px): downloaded ({size_kb:.0f} KB)"
                        )
                        total_downloaded += 1
                    else:
                        total_failed += 1
                    # Be nice to the API
                    time.sleep(0.5)

                if dest.exists():
                    manifest[manifest_key]["files"][size_label] = {
                        "path": str(dest.relative_to(CORPUS_DIR)),
                        "size_bytes": dest.stat().st_size,
                    }

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
    print(f"  Corpus directory: {CORPUS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""Download a curated test corpus organized by technical groups.

Usage:
    python scripts/download_corpus.py
    python scripts/download_corpus.py --group high_res standard
    python scripts/download_corpus.py --dry-run
    python scripts/download_corpus.py --skip-external

Requires UNSPLASH_ACCESS_KEY env var or pass --key.

Downloads images in 4 technical groups:
  - high_res:   3 photos at 2400px (landscape, architecture, texture)
  - standard:   3 photos at 1200px (portrait, food, macro)
  - compact:    3 photos at 400px  (abstract, monochrome, colorful)
  - deep_color: Native AVIF/HEIC/JXL samples from external sources

Each photo is downloaded in 4 CDN-native formats (JPEG, PNG, AVIF, WebP).
Generates groups.json manifest for the benchmark system.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "corpus"

# ---------------------------------------------------------------------------
# Curated photos per group (Unsplash photo IDs)
# ---------------------------------------------------------------------------

GROUP_PHOTOS = {
    "high_res": [
        {"id": "mI02K_LxlfU", "name": "landscape_01", "width": 2400, "content": "landscape"},
        {"id": "KcQokutZS7k", "name": "architecture_02", "width": 2400, "content": "architecture"},
        {"id": "UJzAatnX_tg", "name": "texture_03", "width": 2400, "content": "texture"},
    ],
    "standard": [
        {"id": "fI9R5Aj6UfU", "name": "portrait_01", "width": 1200, "content": "portrait"},
        {"id": "XePlVmD_YcA", "name": "food_02", "width": 1200, "content": "food"},
        {"id": "VcDpd_uF-Y4", "name": "macro_03", "width": 1200, "content": "macro"},
    ],
    "compact": [
        {"id": "7wUorDiCMSU", "name": "abstract_01", "width": 400, "content": "abstract"},
        {"id": "B_Z3TJKEWZs", "name": "monochrome_02", "width": 400, "content": "monochrome"},
        {"id": "iNimytf5qis", "name": "colorful_03", "width": 400, "content": "colorful"},
    ],
}

# CDN formats to download per photo
# (file_extension, CDN fm= param, extra URL params)
CDN_FORMATS = [
    ("jpg", "jpg", "q=90"),
    ("png", "png", ""),
    ("avif", "avif", "q=80"),
    ("webp", "webp", "q=90"),
]

# External native samples for deep_color group
_LINKU_BASE = "https://raw.githubusercontent.com/link-u/avif-sample-images/master"
EXTERNAL_SAMPLES = {
    "avif_native": [
        ("hato_8bit_yuv420", f"{_LINKU_BASE}/hato.profile0.8bpc.yuv420.avif", "avif"),
        ("hato_10bit_yuv422", f"{_LINKU_BASE}/hato.profile2.10bpc.yuv422.avif", "avif"),
        ("hato_12bit_yuv422", f"{_LINKU_BASE}/hato.profile2.12bpc.yuv422.avif", "avif"),
        ("fox_8bit_yuv420", f"{_LINKU_BASE}/fox.profile0.8bpc.yuv420.avif", "avif"),
        ("fox_10bit_yuv444", f"{_LINKU_BASE}/fox.profile1.10bpc.yuv444.avif", "avif"),
        ("fox_12bit_yuv422", f"{_LINKU_BASE}/fox.profile2.12bpc.yuv422.avif", "avif"),
        ("kimono_standard", f"{_LINKU_BASE}/kimono.avif", "avif"),
        ("fox_12bit_yuv420", f"{_LINKU_BASE}/fox.profile2.12bpc.yuv420.avif", "avif"),
    ],
}


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


def build_unsplash_url(photo_id: str, width: int, fmt: str, extra: str) -> str:
    """Build a sized Unsplash CDN URL for a given photo ID."""
    base = f"https://images.unsplash.com/photo-{photo_id}"
    params = f"w={width}&fm={fmt}&fit=crop"
    if extra:
        params += f"&{extra}"
    return f"{base}?{params}"


def fetch_photo_raw_url(photo_id: str, access_key: str) -> str | None:
    """Fetch the raw URL for a photo from the Unsplash API."""
    url = f"https://api.unsplash.com/photos/{photo_id}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Client-ID {access_key}",
            "Accept-Version": "v1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["urls"]["raw"]
    except Exception as e:
        print(f"  WARNING: Could not fetch raw URL for {photo_id}: {e}")
        return None


def build_cdn_url(raw_url: str, width: int, fmt: str, extra: str) -> str:
    """Build a sized CDN URL from a raw Unsplash URL."""
    sep = "&" if "?" in raw_url else "?"
    params = f"w={width}&fm={fmt}&fit=crop"
    if extra:
        params += f"&{extra}"
    return f"{raw_url}{sep}{params}"


def download_group_photos(
    group: str,
    photos: list[dict],
    access_key: str,
    force: bool = False,
    dry_run: bool = False,
    cdn_formats: list[tuple] | None = None,
) -> tuple[int, int, int, list[dict]]:
    """Download photos for a group. Returns (downloaded, skipped, failed, file_infos)."""
    if cdn_formats is None:
        cdn_formats = CDN_FORMATS

    downloaded, skipped, failed = 0, 0, 0
    file_infos = []
    group_dir = CORPUS_DIR / group

    print(f"\n{'='*60}")
    print(f"Group: {group} ({len(photos)} photos)")
    print(f"{'='*60}")

    for photo in photos:
        photo_id = photo["id"]
        name = photo["name"]
        width = photo["width"]

        # Fetch raw URL from API
        if not dry_run:
            raw_url = fetch_photo_raw_url(photo_id, access_key)
            time.sleep(1)  # Rate limiting
        else:
            raw_url = None

        print(f"\n  {name} (id: {photo_id}, {width}px)")

        for ext, fm, extra in cdn_formats:
            filename = f"{name}.{ext}"
            dest = group_dir / filename
            rel_path = f"{group}/{filename}"

            if dry_run:
                print(f"    Would download: {ext.upper()} -> {rel_path}")
                continue

            if dest.exists() and not force:
                size_bytes = dest.stat().st_size
                print(f"    {ext.upper()}: exists ({size_bytes / 1024:.0f} KB)")
                skipped += 1
                file_infos.append(
                    {
                        "path": rel_path,
                        "format": "jpeg" if ext == "jpg" else ext,
                        "source_type": "cdn",
                        "category": _size_category(width),
                        "size_bytes": size_bytes,
                    }
                )
                continue

            if raw_url:
                url = build_cdn_url(raw_url, width, fm, extra)
            else:
                # Fallback to direct URL pattern
                url = build_unsplash_url(photo_id, width, fm, extra)

            ok = download_file(url, dest, force=force)
            if ok:
                size_bytes = dest.stat().st_size
                print(f"    {ext.upper()}: downloaded ({size_bytes / 1024:.0f} KB)")
                downloaded += 1
                file_infos.append(
                    {
                        "path": rel_path,
                        "format": "jpeg" if ext == "jpg" else ext,
                        "source_type": "cdn",
                        "category": _size_category(width),
                        "size_bytes": size_bytes,
                    }
                )
            else:
                failed += 1
            time.sleep(0.5)

    return downloaded, skipped, failed, file_infos


def download_external_samples(
    force: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int, list[dict]]:
    """Download external native samples for deep_color group."""
    downloaded, skipped, failed = 0, 0, 0
    file_infos = []

    print(f"\n{'='*60}")
    print("Group: deep_color (external native samples)")
    print(f"{'='*60}")

    for subdir, samples in EXTERNAL_SAMPLES.items():
        dest_dir = CORPUS_DIR / "deep_color" / subdir
        print(f"\n  {subdir} ({len(samples)} files)")

        for name, url, fmt in samples:
            ext = url.rsplit(".", 1)[-1]
            dest = dest_dir / f"{name}.{ext}"
            rel_path = f"deep_color/{subdir}/{name}.{ext}"

            if dry_run:
                print(f"    Would download: {name} -> {rel_path}")
                continue

            if dest.exists() and not force:
                size_bytes = dest.stat().st_size
                print(f"    {name}: exists ({size_bytes / 1024:.0f} KB)")
                skipped += 1
                file_infos.append(
                    {
                        "path": rel_path,
                        "format": fmt,
                        "source_type": "native",
                        "category": "large",
                        "size_bytes": size_bytes,
                    }
                )
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            ok = download_file(url, dest, force=force)
            if ok:
                size_bytes = dest.stat().st_size
                print(f"    {name}: downloaded ({size_bytes / 1024:.0f} KB)")
                downloaded += 1
                file_infos.append(
                    {
                        "path": rel_path,
                        "format": fmt,
                        "source_type": "native",
                        "category": "large",
                        "size_bytes": size_bytes,
                    }
                )
            else:
                failed += 1
            time.sleep(0.3)

    return downloaded, skipped, failed, file_infos


def _size_category(width: int) -> str:
    """Map download width to size category."""
    if width >= 2000:
        return "large"
    elif width >= 800:
        return "medium"
    return "small"


def generate_groups_json(group_files: dict[str, list[dict]]) -> dict:
    """Generate the groups.json manifest."""
    manifest = {"version": 1, "groups": {}}

    for group_key, files in group_files.items():
        manifest["groups"][group_key] = {
            "files": files,
        }

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download group-based test corpus")
    parser.add_argument(
        "--key",
        default=os.environ.get("UNSPLASH_ACCESS_KEY"),
        help="Unsplash API access key (env: UNSPLASH_ACCESS_KEY)",
    )
    parser.add_argument(
        "--group",
        nargs="+",
        choices=["high_res", "standard", "compact", "deep_color"],
        help="Only download specific groups",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip downloading external native samples (deep_color group)",
    )
    args = parser.parse_args()

    if not args.key and not args.dry_run:
        print("ERROR: Set UNSPLASH_ACCESS_KEY env var or pass --key")
        sys.exit(1)

    access_key = args.key or ""
    groups_to_download = args.group or list(GROUP_PHOTOS.keys()) + ["deep_color"]

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    all_group_files: dict[str, list[dict]] = {}

    # Download CDN photos for each group
    for group_key, photos in GROUP_PHOTOS.items():
        if group_key not in groups_to_download:
            continue
        d, s, f, files = download_group_photos(
            group_key,
            photos,
            access_key,
            force=args.force,
            dry_run=args.dry_run,
        )
        total_downloaded += d
        total_skipped += s
        total_failed += f
        all_group_files[group_key] = files

    # Download external native samples for deep_color
    if "deep_color" in groups_to_download and not args.skip_external:
        d, s, f, files = download_external_samples(
            force=args.force,
            dry_run=args.dry_run,
        )
        total_downloaded += d
        total_skipped += s
        total_failed += f
        all_group_files["deep_color"] = files

    # Generate groups.json manifest
    if not args.dry_run and all_group_files:
        manifest = generate_groups_json(all_group_files)
        manifest_path = CORPUS_DIR / "groups.json"

        # Merge with existing manifest if present
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            for group_key, group_data in existing.get("groups", {}).items():
                if group_key not in manifest["groups"]:
                    manifest["groups"][group_key] = group_data

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nManifest saved to {manifest_path}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Skipped (exists): {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Groups: {', '.join(groups_to_download)}")
    print(f"  Corpus directory: {CORPUS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

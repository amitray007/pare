"""Corpus management for benchmark groups.

Loads groups.json manifest and provides group-aware case selection.
Falls back to dimension-based classification when groups.json is absent.
"""

import io
import json
from pathlib import Path

from PIL import Image

from benchmarks.cases import BenchmarkCase

# File extension to format mapping
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


def load_groups_manifest(corpus_dir: Path) -> dict | None:
    """Load groups.json from corpus directory. Returns None if not found."""
    manifest_path = corpus_dir / "groups.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _classify_group_by_dims(data: bytes, fmt: str, filepath: Path) -> str:
    """Classify an image into a group based on dimensions and path."""
    # Deep-color: files in specific native directories
    parts = filepath.parts
    if "deep_color" in parts or "avif_native" in parts:
        return "deep_color"

    if fmt in ("svg", "svgz"):
        return "standard"

    try:
        img = Image.open(io.BytesIO(data))
        max_dim = max(img.size)
        file_size = len(data)

        if max_dim >= 2000 and file_size > 500_000:
            return "high_res"
        elif max_dim < 800:
            return "compact"
        else:
            return "standard"
    except Exception:
        # Fallback to file size
        size = len(data)
        if size > 500_000:
            return "high_res"
        if size < 100_000:
            return "compact"
        return "standard"


def load_corpus_cases(
    corpus_dir: Path,
    groups: list[str] | None = None,
    formats: list[str] | None = None,
) -> list[BenchmarkCase]:
    """Load corpus images as BenchmarkCase list, optionally filtered by group and format.

    If groups.json exists, uses it for group assignment.
    Otherwise falls back to dimension-based classification.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        return []

    manifest = load_groups_manifest(corpus_path)
    cases = []

    if manifest and "groups" in manifest:
        # Manifest-driven loading
        for group_key, group_data in manifest["groups"].items():
            if groups and group_key not in groups:
                continue
            for file_info in group_data.get("files", []):
                fmt = file_info.get("format", "")
                if formats and fmt not in formats:
                    continue
                filepath = corpus_path / file_info["path"]
                if not filepath.exists():
                    continue
                data = filepath.read_bytes()
                cases.append(
                    BenchmarkCase(
                        name=f"{group_key}/{filepath.stem}",
                        data=data,
                        fmt=fmt,
                        category=file_info.get("category", "medium"),
                        content=filepath.parent.name,
                        group=group_key,
                    )
                )
    else:
        # Fallback: scan directory and classify by dimensions
        for filepath in sorted(corpus_path.rglob("*")):
            if not filepath.is_file():
                continue
            ext = filepath.suffix.lower()
            fmt = _EXT_TO_FMT.get(ext)
            if fmt is None:
                continue
            if formats and fmt not in formats:
                continue

            data = filepath.read_bytes()
            group = _classify_group_by_dims(data, fmt, filepath)

            if groups and group not in groups:
                continue

            cases.append(
                BenchmarkCase(
                    name=f"{filepath.parent.name}/{filepath.stem}",
                    data=data,
                    fmt=fmt,
                    category=group,
                    content=filepath.parent.name,
                    group=group,
                )
            )

    return cases


def scan_corpus_by_group(
    corpus_dir: Path,
) -> dict[str, dict[str, list[Path]]]:
    """Scan corpus and group files by group and format.

    Returns: {group_key: {format: [paths]}}
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        return {}

    manifest = load_groups_manifest(corpus_path)
    result: dict[str, dict[str, list[Path]]] = {}

    if manifest and "groups" in manifest:
        for group_key, group_data in manifest["groups"].items():
            for file_info in group_data.get("files", []):
                fmt = file_info.get("format", "")
                filepath = corpus_path / file_info["path"]
                if filepath.exists():
                    result.setdefault(group_key, {}).setdefault(fmt, []).append(filepath)
    else:
        # Fallback scan
        for filepath in sorted(corpus_path.rglob("*")):
            if not filepath.is_file():
                continue
            ext = filepath.suffix.lower()
            fmt = _EXT_TO_FMT.get(ext)
            if fmt is None:
                continue
            data = filepath.read_bytes()
            group = _classify_group_by_dims(data, fmt, filepath)
            result.setdefault(group, {}).setdefault(fmt, []).append(filepath)

    return result

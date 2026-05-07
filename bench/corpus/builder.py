"""Build, refresh, and verify a corpus on disk.

The builder walks a manifest, synthesizes each entry, encodes into every
declared `output_format`, and writes the result with an atomic rename.
Existence-based skip: if the target file already exists on disk and
`--force` is not set, the entry's encode step is skipped — fast for
incremental rebuilds, but does NOT validate that the on-disk bytes
match the current manifest. Use `bench.corpus verify` (which re-runs
the synthesizer and checks `expected_pixel_sha256`) to catch stale
files; `--force` re-synthesizes and re-encodes everything.

Bucket validation runs after every encode: if the encoded file lands
outside the entry's declared bucket, the build fails with an actionable
error rather than silently producing a misleading corpus.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageSequence

from bench.corpus.conversion import (
    FormatNotSupportedError,
    encode,
    is_animation_format,
    supported_formats,
)
from bench.corpus.fetchers import DEFAULT_CACHE_ROOT, fetch
from bench.corpus.manifest import (
    _VECTOR_FORMATS,
    Manifest,
    ManifestEntry,
    Synthesized,
    bucket_for_size,
    is_vector_entry,
    pixel_sha256,
)
from bench.corpus.sizing import in_bucket
from bench.corpus.synthesis import synthesize

logger = logging.getLogger(__name__)


@dataclass
class BuildOutcome:
    written: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    bucket_violations: list[str] = field(default_factory=list)
    format_skipped: list[str] = field(default_factory=list)
    pixel_hashes: dict[str, str] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    byte_hashes: dict[str, str] = field(default_factory=dict)  # {entry_name: sha256} for vectors

    @property
    def ok(self) -> bool:
        return not self.bucket_violations


def file_path(corpus_root: Path, entry: ManifestEntry, fmt: str) -> Path:
    """Layout: <root>/<bucket>/<format>/<name>.<ext>."""
    ext = "apng" if fmt == "apng" else fmt
    return corpus_root / entry.bucket.value / fmt / f"{entry.name}.{ext}"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_fetched_content(entry: ManifestEntry, cache_root: Path) -> Synthesized:
    """Fetch and decode a fetched-photo entry, returning PIL content.

    For animated images, returns a list of frames.  For static images,
    returns a single Image loaded into memory (so the file handle and
    BytesIO can be GC'd).
    """
    path = fetch(entry.source, cache_root)  # type: ignore[arg-type]
    data = path.read_bytes()
    img = Image.open(BytesIO(data))
    if getattr(img, "is_animated", False):
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
        return frames
    img.load()  # ensure pixels are decoded before the BytesIO goes out of scope
    return img.copy()


def _load_vector_bytes(entry: ManifestEntry, cache_root: Path) -> bytes:
    """Fetch a vector entry and return the raw source bytes.

    No Image.open() — SVG/SVGZ sources are XML, not raster pixels.
    The fetcher verifies the SHA-256 of the downloaded bytes before returning.
    """
    path = fetch(entry.source, cache_root)  # type: ignore[arg-type]
    return path.read_bytes()


def _check_no_mixed_vector_raster(entry: ManifestEntry) -> str | None:
    """Return an error string if the entry mixes vector and raster output formats.

    Mixed entries are forbidden: if any output_format is a vector format, ALL
    output_formats must be vector formats.  This prevents confusing code paths
    where the same content would be routed through both Image.open() and
    pass-through paths.
    """
    has_vector = any(fmt in _VECTOR_FORMATS for fmt in entry.output_formats)
    has_raster = any(fmt not in _VECTOR_FORMATS for fmt in entry.output_formats)
    if has_vector and has_raster:
        vector_fmts = [f for f in entry.output_formats if f in _VECTOR_FORMATS]
        raster_fmts = [f for f in entry.output_formats if f not in _VECTOR_FORMATS]
        return (
            f"{entry.name}: output_formats mixes vector {vector_fmts} and raster {raster_fmts}. "
            f"Vector and raster formats cannot share an entry — split into separate entries."
        )
    return None


def build(
    manifest: Manifest,
    corpus_root: Path,
    *,
    force: bool = False,
    formats_filter: set[str] | None = None,
    bucket_filter: str | None = None,
    tag_filter: str | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> BuildOutcome:
    """Synthesize (or fetch) and encode every entry in the manifest."""
    available = set(supported_formats())
    outcome = BuildOutcome()

    entries = manifest.filter(bucket=bucket_filter, tag=tag_filter)

    for entry in entries:
        # Reject mixed vector+raster entries early
        mix_error = _check_no_mixed_vector_raster(entry)
        if mix_error:
            outcome.bucket_violations.append(mix_error)
            continue

        if is_vector_entry(entry):
            # Vector path: fetch raw bytes (fetched) or synthesize bytes (synthetic).
            # In both cases the result must be bytes — no Image.open() involved.
            if entry.source is not None:
                # Fetched vector entry — download from declared URL.
                try:
                    content: Synthesized = _load_vector_bytes(entry, cache_root)
                    outcome.source_hashes[entry.name] = entry.source.sha256
                    outcome.byte_hashes[entry.name] = hashlib.sha256(content).hexdigest()  # type: ignore[arg-type]
                except Exception as e:
                    outcome.bucket_violations.append(f"{entry.name}: vector fetch failed: {e}")
                    continue
            else:
                # Synthesized vector entry — call the registered synthesizer which
                # returns bytes directly (no Pillow involved).
                try:
                    content = synthesize(entry)
                    if not isinstance(content, (bytes, bytearray)):
                        raise TypeError(
                            f"vector synthesizer for {entry.content_kind!r} returned "
                            f"{type(content).__name__!r}; expected bytes"
                        )
                    outcome.byte_hashes[entry.name] = hashlib.sha256(content).hexdigest()  # type: ignore[arg-type]
                except Exception as e:
                    outcome.bucket_violations.append(f"{entry.name}: vector synthesis failed: {e}")
                    continue
            # No pixel_sha256 for vector entries
        elif entry.source is not None:
            # Fetched raster entry — download and decode from source URL
            try:
                content = _load_fetched_content(entry, cache_root)
                outcome.source_hashes[entry.name] = entry.source.sha256
            except Exception as e:
                outcome.bucket_violations.append(f"{entry.name}: fetch failed: {e}")
                continue
            outcome.pixel_hashes[entry.name] = pixel_sha256(content)
        else:
            # Synthesized raster entry
            try:
                content = synthesize(entry)
            except Exception as e:
                outcome.bucket_violations.append(f"{entry.name}: synthesis failed: {e}")
                continue
            outcome.pixel_hashes[entry.name] = pixel_sha256(content)

        for fmt in entry.output_formats:
            if formats_filter and fmt not in formats_filter:
                continue
            if fmt not in available:
                outcome.format_skipped.append(f"{entry.name}.{fmt} (encoder unavailable)")
                continue

            target = file_path(corpus_root, entry, fmt)
            if not force and target.exists():
                outcome.skipped.append(target)
                continue

            try:
                blob = encode(content, fmt)
            except FormatNotSupportedError as e:
                outcome.format_skipped.append(f"{entry.name}.{fmt}: {e}")
                continue
            except Exception as e:
                outcome.bucket_violations.append(f"{entry.name}.{fmt}: encode failed: {e}")
                continue

            actual_bucket = bucket_for_size(len(blob))
            if not in_bucket(len(blob), entry.bucket):
                outcome.bucket_violations.append(
                    f"{entry.name}.{fmt}: bucket violation — encoded {len(blob)}B "
                    f"falls in {actual_bucket.value!r}, but entry declares {entry.bucket.value!r}"
                )
                continue

            _atomic_write_bytes(target, blob)
            outcome.written.append(target)

    return outcome


def reseal_manifest(
    manifest: Manifest,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Manifest:
    """Run synthesis (or fetch) once per entry and write current hashes back.

    For raster entries: populates `expected_pixel_sha256`.
    For vector entries: populates `expected_byte_sha256` with the source SHA.

    Output is a new Manifest object — caller is responsible for writing it
    back to disk.
    """
    sealed = Manifest(
        name=manifest.name,
        manifest_version=manifest.manifest_version,
        library_versions=dict(manifest.library_versions),
        entries=[],
    )
    for entry in manifest.entries:
        if is_vector_entry(entry):
            # Vector entry: store byte-level SHA of the source bytes.
            # Supports both fetched (entry.source set) and synthesized (source None) entries.
            if entry.source is not None:
                raw_bytes = _load_vector_bytes(entry, cache_root)
            else:
                raw_bytes = synthesize(entry)  # type: ignore[assignment]
                if not isinstance(raw_bytes, (bytes, bytearray)):
                    raise TypeError(
                        f"vector synthesizer for {entry.content_kind!r} returned "
                        f"{type(raw_bytes).__name__!r}; expected bytes"
                    )
            byte_sha = hashlib.sha256(raw_bytes).hexdigest()
            sealed_entry = ManifestEntry(
                name=entry.name,
                bucket=entry.bucket,
                content_kind=entry.content_kind,
                seed=entry.seed,
                width=entry.width,
                height=entry.height,
                output_formats=list(entry.output_formats),
                params=dict(entry.params),
                tags=list(entry.tags),
                expected_pixel_sha256=None,  # not applicable for vector
                encoded_sha256=dict(entry.encoded_sha256),
                source=entry.source,
                expected_byte_sha256={"source": byte_sha},
                bit_depth=entry.bit_depth,
            )
        else:
            if entry.source is not None:
                content = _load_fetched_content(entry, cache_root)
            else:
                content = synthesize(entry)
            sealed_entry = ManifestEntry(
                name=entry.name,
                bucket=entry.bucket,
                content_kind=entry.content_kind,
                seed=entry.seed,
                width=entry.width,
                height=entry.height,
                output_formats=list(entry.output_formats),
                params=dict(entry.params),
                tags=list(entry.tags),
                expected_pixel_sha256=pixel_sha256(content),
                encoded_sha256=dict(entry.encoded_sha256),
                source=entry.source,
                expected_byte_sha256=None,
                bit_depth=entry.bit_depth,
            )
        sealed.entries.append(sealed_entry)
    return sealed


__all__ = [
    "BuildOutcome",
    "build",
    "file_path",
    "is_animation_format",
    "reseal_manifest",
    "DEFAULT_CACHE_ROOT",
]

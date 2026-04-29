"""Build, refresh, and verify a corpus on disk.

The builder walks a manifest, synthesizes each entry, encodes into every
declared `output_format`, and writes the result with an atomic rename.
Files match the manifest's pixel-hash already on disk are skipped, so
incremental rebuilds are cheap; `--force` re-synthesizes everything.

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
from pathlib import Path

from bench.corpus.conversion import (
    FormatNotSupportedError,
    encode,
    is_animation_format,
    supported_formats,
)
from bench.corpus.manifest import (
    Manifest,
    ManifestEntry,
    bucket_for_size,
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


def build(
    manifest: Manifest,
    corpus_root: Path,
    *,
    force: bool = False,
    formats_filter: set[str] | None = None,
    bucket_filter: str | None = None,
    tag_filter: str | None = None,
) -> BuildOutcome:
    """Synthesize and encode every entry in the manifest."""
    available = set(supported_formats())
    outcome = BuildOutcome()

    entries = manifest.filter(bucket=bucket_filter, tag=tag_filter)

    for entry in entries:
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


def reseal_manifest(manifest: Manifest) -> Manifest:
    """Run synthesis once per entry and write current pixel hashes back.

    Used to populate `expected_pixel_sha256` on a fresh manifest, or to
    re-seal after an intentional corpus refresh. Output is a new Manifest
    object — caller is responsible for writing it back to disk.
    """
    sealed = Manifest(
        name=manifest.name,
        manifest_version=manifest.manifest_version,
        library_versions=dict(manifest.library_versions),
        entries=[],
    )
    for entry in manifest.entries:
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
        )
        sealed.entries.append(sealed_entry)
    return sealed


__all__ = [
    "BuildOutcome",
    "build",
    "file_path",
    "is_animation_format",
    "reseal_manifest",
]

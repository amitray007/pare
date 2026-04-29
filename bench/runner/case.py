"""Case loading: turn a manifest + corpus dir into runnable Cases.

A `Case` is a single (manifest entry × output_format × preset) cell. The
benchmark runner iterates Cases, measuring each one. Case IDs follow
`<entry_name>.<format>@<preset>` so any reporter can sort, filter, or
compare runs without parsing nested dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bench.corpus.builder import file_path
from bench.corpus.manifest import Manifest, ManifestEntry

# Preset to quality mapping, mirroring `estimation.presets.PRESETS_BY_NAME`
# so the bench measures what production /estimate actually requests.
# Lower = more aggressive lossy.
PRESET_QUALITY: dict[str, int] = {
    "high": 40,
    "medium": 60,
    "low": 75,
}

DEFAULT_PRESETS: tuple[str, ...] = ("high", "medium", "low")


@dataclass
class Case:
    """One (entry × format × preset) execution unit."""

    case_id: str
    name: str
    bucket: str
    fmt: str
    preset: str
    quality: int
    file_path: Path
    input_size: int

    def load(self) -> bytes:
        return self.file_path.read_bytes()


class CorpusFileMissing(FileNotFoundError):
    """A manifest entry's encoded file is not present on disk."""


def _make_case(
    entry: ManifestEntry,
    fmt: str,
    preset: str,
    corpus_root: Path,
) -> Case:
    quality = PRESET_QUALITY[preset]
    path = file_path(corpus_root, entry, fmt)
    if not path.exists():
        raise CorpusFileMissing(
            f"corpus file missing: {path}. "
            f"Run `python -m bench.corpus build --manifest {corpus_root.parent.name or 'core'}`."
        )
    return Case(
        case_id=f"{entry.name}.{fmt}@{preset}",
        name=entry.name,
        bucket=entry.bucket.value,
        fmt=fmt,
        preset=preset,
        quality=quality,
        file_path=path,
        input_size=path.stat().st_size,
    )


def load_cases(
    manifest: Manifest,
    corpus_root: Path,
    *,
    fmt_filter: set[str] | None = None,
    bucket_filter: str | None = None,
    tag_filter: str | None = None,
    preset_filter: set[str] | None = None,
) -> list[Case]:
    """Build a list of Cases from a manifest and an on-disk corpus.

    Filters are applied additively. `preset_filter` defaults to all
    three presets. Missing corpus files raise `CorpusFileMissing` —
    fail fast rather than silently skip; an incomplete corpus is a
    setup error.
    """
    presets = preset_filter or set(DEFAULT_PRESETS)
    unknown = presets - PRESET_QUALITY.keys()
    if unknown:
        raise ValueError(f"unknown presets: {sorted(unknown)}")

    cases: list[Case] = []
    for entry in manifest.filter(bucket=bucket_filter, tag=tag_filter):
        for fmt in entry.output_formats:
            if fmt_filter and fmt not in fmt_filter:
                continue
            for preset in presets:
                cases.append(_make_case(entry, fmt, preset, corpus_root))
    return cases

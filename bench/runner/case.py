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
            f"Run `python -m bench.corpus build` against the matching --manifest "
            f"(e.g. `--manifest core` or `--manifest full`)."
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
    exclude_tag: str | None = None,
    preset_filter: set[str] | None = None,
    skip_missing: bool = True,
) -> list[Case]:
    """Build a list of Cases from a manifest and an on-disk corpus.

    Filters are applied additively. `preset_filter` defaults to all
    three presets.

    `exclude_tag`: if set, entries that carry this tag are skipped. This
    is additive (does not change behaviour when omitted). Primary use case:
    ``exclude_tag="fat_input"`` to keep timing/quick/memory runs cheap while
    still keeping fat-input entries in the manifest for explicit load testing.

    Missing-file handling depends on `skip_missing`:
      - `True` (default): log a warning and skip cases whose encoded
        file is absent. This is the right default in environments
        where some formats are intentionally unavailable (e.g. JXL
        in the production Docker image without libjxl, AVIF without
        pillow_avif). The build step's `format_skipped` accounting
        already records the gap; failing the runner on the same
        gap would just double-report.
      - `False`: raise `CorpusFileMissing` on the first absent file.
        Useful for tests that want to guarantee a complete corpus.
    """
    import logging

    log = logging.getLogger(__name__)
    presets = preset_filter or set(DEFAULT_PRESETS)
    unknown = presets - PRESET_QUALITY.keys()
    if unknown:
        raise ValueError(f"unknown presets: {sorted(unknown)}")

    cases: list[Case] = []
    skipped: list[str] = []
    for entry in manifest.filter(bucket=bucket_filter, tag=tag_filter):
        if exclude_tag and exclude_tag in entry.tags:
            continue
        for fmt in entry.output_formats:
            if fmt_filter and fmt not in fmt_filter:
                continue
            for preset in presets:
                try:
                    cases.append(_make_case(entry, fmt, preset, corpus_root))
                except CorpusFileMissing:
                    if not skip_missing:
                        raise
                    skipped.append(f"{entry.name}.{fmt}@{preset}")
    if skipped:
        log.warning(
            "load_cases skipped %d case(s) with missing corpus files "
            "(format unavailable in this env): %s%s",
            len(skipped),
            ", ".join(skipped[:5]),
            f", … (+{len(skipped) - 5} more)" if len(skipped) > 5 else "",
        )
    return cases

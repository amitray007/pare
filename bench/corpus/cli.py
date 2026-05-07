"""Command-line interface for the corpus builder.

    python -m bench.corpus build   [--manifest core] [--bucket B] [--fmt F] [--tag T] [--force] [--seal]
    python -m bench.corpus verify  [--manifest core]
    python -m bench.corpus list    [--manifest core]

Exit codes (verify):
    0  pass
    1  pixel-hash mismatch
    2  manifest had entries with no expected_pixel_sha256 (run build --seal)
    3  schema or synthesis error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bench.corpus.builder import build, reseal_manifest
from bench.corpus.conversion import supported_formats
from bench.corpus.fetchers import DEFAULT_CACHE_ROOT, FetchError, fetch
from bench.corpus.manifest import (
    Manifest,
    ManifestSchemaError,
    collect_library_versions,
    verify,
)
from bench.corpus.synthesis import synthesize

DEFAULT_CORPUS_ROOT = Path("tests/corpus")
DEFAULT_MANIFEST_DIR = Path(__file__).parent / "manifests"

logger = logging.getLogger("bench.corpus")


def _manifest_path(name: str) -> Path:
    if name.endswith(".json"):
        return Path(name)
    return DEFAULT_MANIFEST_DIR / f"{name}.json"


def _load_manifest(name: str) -> Manifest:
    path = _manifest_path(name)
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    return Manifest.load(path)


def cmd_build(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    corpus_root = Path(args.out)
    formats_filter = set(args.fmt) if args.fmt else None
    cache_root = Path(args.cache) if args.cache else DEFAULT_CACHE_ROOT

    if args.seal:
        sealed = reseal_manifest(manifest, cache_root=cache_root)
        sealed.library_versions = collect_library_versions()
        out_path = _manifest_path(args.manifest)
        sealed.save(out_path)
        logger.info("sealed manifest -> %s (%d entries)", out_path, len(sealed.entries))
        manifest = sealed

    outcome = build(
        manifest,
        corpus_root,
        force=args.force,
        formats_filter=formats_filter,
        bucket_filter=args.bucket,
        tag_filter=args.tag,
        cache_root=cache_root,
    )

    print(
        f"corpus={corpus_root} written={len(outcome.written)} "
        f"skipped={len(outcome.skipped)} format_skipped={len(outcome.format_skipped)} "
        f"bucket_violations={len(outcome.bucket_violations)}"
    )
    if outcome.format_skipped:
        for line in outcome.format_skipped:
            logger.warning("skipped: %s", line)
    for line in outcome.bucket_violations:
        print(f"BUCKET VIOLATION: {line}", file=sys.stderr)

    return 0 if outcome.ok else 1


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        manifest = _load_manifest(args.manifest)
    except ManifestSchemaError as e:
        print(f"schema error: {e}", file=sys.stderr)
        return 3

    cache_root = DEFAULT_CACHE_ROOT

    def _synthesize_or_fetch(entry):
        """Dispatch to fetcher/vector-loader or synthesizer depending on entry type."""
        from bench.corpus.manifest import is_vector_entry

        if is_vector_entry(entry):
            if entry.source is not None:
                # Fetched vector (e.g. fetched_vector) — load from URL cache.
                from bench.corpus.builder import _load_vector_bytes

                return _load_vector_bytes(entry, cache_root)
            # Synthesized vector (e.g. vector_geometric) — call the synthesizer
            # which returns bytes directly.
            return synthesize(entry)
        if entry.source is not None:
            from bench.corpus.builder import _load_fetched_content

            return _load_fetched_content(entry, cache_root)
        return synthesize(entry)

    result = verify(manifest, _synthesize_or_fetch)

    if result.schema_errors:
        print("SCHEMA ERRORS:", file=sys.stderr)
        for line in result.schema_errors:
            print(f"  {line}", file=sys.stderr)
    if result.missing:
        print(
            "MISSING (no expected_pixel_sha256 / expected_byte_sha256 — run `build --seal`):",
            file=sys.stderr,
        )
        for line in result.missing:
            print(f"  {line}", file=sys.stderr)
    if result.mismatches:
        print("PIXEL MISMATCHES:", file=sys.stderr)
        for line in result.mismatches:
            print(f"  {line}", file=sys.stderr)

    if result.ok:
        print(f"verify OK: {len(manifest.entries)} entries")
    return result.exit_code


def cmd_list(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    print(f"manifest={manifest.name} entries={len(manifest.entries)}")
    print(f"supported_formats={','.join(supported_formats())}")
    print()
    print(f"{'name':<32} {'bucket':<7} {'kind':<22} {'dims':<11} {'fmts':<24} tags")
    print("-" * 110)
    for e in manifest.entries:
        dims = f"{e.width}x{e.height}"
        fmts = ",".join(e.output_formats)
        tags = ",".join(e.tags) if e.tags else "-"
        print(f"{e.name:<32} {e.bucket.value:<7} {e.content_kind:<22} {dims:<11} {fmts:<24} {tags}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Pre-warm the fetcher cache for every fetched entry in the manifest."""
    manifest = _load_manifest(args.manifest)
    cache_root = Path(args.cache) if args.cache else DEFAULT_CACHE_ROOT

    fetched_entries = [e for e in manifest.entries if e.source is not None]
    if not fetched_entries:
        print(f"manifest={manifest.name}: no fetched entries")
        return 0

    successes = 0
    failures = 0
    for entry in fetched_entries:
        try:
            path = fetch(entry.source, cache_root)  # type: ignore[arg-type]
            print(f"  ok  {entry.name} -> {path}")
            successes += 1
        except FetchError as e:
            print(f"  FAIL {entry.name}: {e}", file=sys.stderr)
            failures += 1

    print(f"\nfetch complete: {successes} ok, {failures} failed")
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.corpus",
        description="Build, verify, and list deterministic image corpora.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable INFO logging")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="synthesize + encode the corpus to disk")
    p_build.add_argument("--manifest", default="core")
    p_build.add_argument("--out", default=str(DEFAULT_CORPUS_ROOT))
    p_build.add_argument("--bucket", default=None)
    p_build.add_argument("--fmt", action="append", default=None)
    p_build.add_argument("--tag", default=None)
    p_build.add_argument("--force", action="store_true")
    p_build.add_argument(
        "--seal",
        action="store_true",
        help="re-synthesize and write expected_pixel_sha256 back into the manifest",
    )
    p_build.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help=f"fetcher cache directory (default: {DEFAULT_CACHE_ROOT})",
    )
    p_build.set_defaults(func=cmd_build)

    p_verify = sub.add_parser("verify", help="re-synthesize and check pixel hashes")
    p_verify.add_argument("--manifest", default="core")
    p_verify.set_defaults(func=cmd_verify)

    p_list = sub.add_parser("list", help="show manifest contents")
    p_list.add_argument("--manifest", default="core")
    p_list.set_defaults(func=cmd_list)

    p_fetch = sub.add_parser("fetch", help="pre-warm fetcher cache for all fetched entries")
    p_fetch.add_argument("--manifest", default="full")
    p_fetch.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help=f"fetcher cache directory (default: {DEFAULT_CACHE_ROOT})",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)

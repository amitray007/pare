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

    if args.seal:
        sealed = reseal_manifest(manifest)
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

    result = verify(manifest, synthesize)

    if result.schema_errors:
        print("SCHEMA ERRORS:", file=sys.stderr)
        for line in result.schema_errors:
            print(f"  {line}", file=sys.stderr)
    if result.missing:
        print("MISSING (no expected_pixel_sha256 — run `build --seal`):", file=sys.stderr)
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
    p_build.set_defaults(func=cmd_build)

    p_verify = sub.add_parser("verify", help="re-synthesize and check pixel hashes")
    p_verify.add_argument("--manifest", default="core")
    p_verify.set_defaults(func=cmd_verify)

    p_list = sub.add_parser("list", help="show manifest contents")
    p_list.add_argument("--manifest", default="core")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)

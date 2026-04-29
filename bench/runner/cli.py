"""Command-line interface for `bench.run` and `bench.compare`.

    python -m bench.run [--mode quick|timing|memory] [--manifest core] \\
                        [--out reports/x.json] [--corpus PATH] \\
                        [--bucket B] [--fmt F] [--tag T] [--preset P] \\
                        [--repeat N] [--warmup N] [--seed N] [--no-shuffle] \\
                        [--annotate KEY=VAL ...]

    python -m bench.compare A.json B.json [--threshold-pct N] [--alpha A]
    python -m bench.report RUN.json [--format markdown|json]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bench.corpus.cli import _load_manifest, _manifest_path
from bench.runner.case import DEFAULT_PRESETS, load_cases
from bench.runner.compare import compare, render_compare_markdown
from bench.runner.modes.memory import run_memory_sync
from bench.runner.modes.quick import run_quick_sync
from bench.runner.modes.timing import run_timing_sync
from bench.runner.report.json_writer import (
    RunMetadata,
    detect_git_info,
    load_run,
    manifest_sha256,
    write_run,
)
from bench.runner.report.markdown import render_run

DEFAULT_CORPUS_ROOT = Path("tests/corpus")

logger = logging.getLogger("bench.runner")


def _parse_annotations(pairs: list[str]) -> dict[str, str]:
    out = {}
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(f"--annotate expects KEY=VAL, got {raw!r}")
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_run(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    corpus_root = Path(args.corpus)
    cases = load_cases(
        manifest,
        corpus_root,
        fmt_filter=set(args.fmt) if args.fmt else None,
        bucket_filter=args.bucket,
        tag_filter=args.tag,
        preset_filter=set(args.preset) if args.preset else None,
    )
    if not cases:
        print("no cases match the given filters", file=sys.stderr)
        return 1

    print(f"running {args.mode} on {len(cases)} case(s)…", file=sys.stderr)

    if args.mode == "quick":
        config = {"warmup": 0, "repeat": 1}
        iterations = run_quick_sync(cases)
    elif args.mode == "memory":
        config = {"warmup": 0, "repeat": 1, "tracemalloc": True}
        iterations = run_memory_sync(cases)
    else:  # timing
        config = {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "shuffle": not args.no_shuffle,
        }
        iterations = run_timing_sync(
            cases,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            shuffle=not args.no_shuffle,
        )

    annotations = _parse_annotations(args.annotate or [])

    manifest_path = _manifest_path(args.manifest)
    metadata = RunMetadata(
        mode=args.mode,
        config=config,
        annotations=annotations,
        manifest_name=manifest.name,
        manifest_sha256=manifest_sha256(manifest_path),
        git=detect_git_info(),
    )

    out_path = Path(args.out)
    write_run(metadata, iterations, out_path)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    result = compare(
        Path(args.baseline),
        Path(args.head),
        threshold_pct=args.threshold_pct,
        alpha=args.alpha,
    )
    if args.format == "markdown":
        print(render_compare_markdown(result))
    else:
        # JSON for piping into automation.
        import json
        from dataclasses import asdict

        print(
            json.dumps(
                {
                    "regressions": [asdict(d) for d in result.regressions],
                    "improvements": [asdict(d) for d in result.improvements],
                    "all": [asdict(d) for d in result.diffs],
                    "only_in_a": result.only_in_a,
                    "only_in_b": result.only_in_b,
                },
                indent=2,
            )
        )
    return result.exit_code


def cmd_report(args: argparse.Namespace) -> int:
    run = load_run(Path(args.path))
    if args.format == "markdown":
        print(render_run(run))
    else:
        import json

        print(json.dumps(run, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.run")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # Default subcommand: `run`. Allow `python -m bench.run --mode timing`
    # without an explicit `run` subcommand.
    p_run = sub.add_parser("run", help="execute a benchmark run")
    p_run.add_argument("--mode", choices=("quick", "timing", "memory"), default="timing")
    p_run.add_argument("--manifest", default="core")
    p_run.add_argument("--corpus", default=str(DEFAULT_CORPUS_ROOT))
    p_run.add_argument("--out", default="reports/bench.json")
    p_run.add_argument("--bucket", default=None)
    p_run.add_argument("--fmt", action="append", default=None)
    p_run.add_argument("--tag", default=None)
    p_run.add_argument(
        "--preset",
        action="append",
        default=None,
        help=f"one of {DEFAULT_PRESETS}; repeatable",
    )
    p_run.add_argument("--warmup", type=int, default=1)
    p_run.add_argument("--repeat", type=int, default=5)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--no-shuffle", action="store_true")
    p_run.add_argument("--annotate", action="append", default=None)
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="diff two runs with significance tests")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("head")
    p_cmp.add_argument("--threshold-pct", type=float, default=10.0)
    p_cmp.add_argument("--alpha", type=float, default=0.05)
    p_cmp.add_argument("--format", choices=("markdown", "json"), default="markdown")
    p_cmp.set_defaults(func=cmd_compare)

    p_rep = sub.add_parser("report", help="render a run as markdown or JSON")
    p_rep.add_argument("path")
    p_rep.add_argument("--format", choices=("markdown", "json"), default="markdown")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(argv) if argv is not None else list(sys.argv[1:])

    # `python -m bench.run --mode timing ...` (no subcommand) defaults to
    # the `run` subcommand. Prepend it before argparse runs so unknown
    # flags don't error out as bad subcommand choices.
    subcommands = {"run", "compare", "report"}
    if not raw or (raw[0] not in subcommands and raw[0] not in {"-h", "--help"}):
        raw = ["run"] + raw

    args = parser.parse_args(raw)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)

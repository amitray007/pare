"""Command-line interface for `bench.run` and `bench.compare`.

    python -m bench.run [--mode quick|timing|memory|accuracy|quality|load] \\
                        [--manifest core] [--out reports/x.json] [--corpus PATH] \\
                        [--bucket B] [--fmt F] [--tag T] [--preset P] \\
                        [--repeat N] [--warmup N] [--seed N] [--no-shuffle] \\
                        [--isolate] [--quality-fast] \\
                        [--n-concurrent N] [--semaphore-size N] [--queue-depth N] \\
                        [--memory-budget-mb N] [--annotate KEY=VAL ...]

    python -m bench.compare A.json B.json [--threshold-pct N] [--alpha A]
                            [--allow-mismatched-mode] [--allow-mismatched-isolate]
                            [--allow-mismatched-platform]
    python -m bench.run report RUN.json [--format markdown|json]

Exit codes for `compare`:
    0  no significant regression
    1  regression flagged in at least one case
    2  schema error (mismatched schema_version, missing required fields) or
       comparability error (mode mismatch between runs)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from bench.corpus.cli import _load_manifest, _manifest_path
from bench.runner.case import DEFAULT_PRESETS, load_cases
from bench.runner.compare import ModeMismatchError, compare, render_compare_markdown
from bench.runner.modes.accuracy import run_accuracy_sync
from bench.runner.modes.load import run_load_sync
from bench.runner.modes.memory import run_memory_sync
from bench.runner.modes.pr import run_pr_sync
from bench.runner.modes.quality import run_quality_sync
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
        exclude_tag=args.exclude_tag,
        preset_filter=set(args.preset) if args.preset else None,
    )
    if not cases:
        print("no cases match the given filters", file=sys.stderr)
        return 1

    print(f"running {args.mode} on {len(cases)} case(s)…", file=sys.stderr)

    # Preflight: warn when JXL is disabled but the required tools are present.
    # This saves the next dev from chasing "UnsupportedFormatError: Format jxl is not enabled"
    # when cjxl and jxlpy are already installed.
    from config import settings

    if not settings.enable_jxl and (args.fmt is None or "jxl" in (args.fmt or [])):
        _cjxl_present = False
        _jxlpy_present = False
        try:
            import shutil

            _cjxl_present = shutil.which("cjxl") is not None
        except Exception:
            pass
        try:
            import jxlpy  # noqa: F401

            _jxlpy_present = True
        except ImportError:
            pass
        if _cjxl_present and _jxlpy_present:
            print(
                "warning: enable_jxl=False but cjxl and jxlpy are both available. "
                "JXL bench cases will fail with UnsupportedFormatError. "
                "Set ENABLE_JXL=true (see .envrc.example) to enable JXL locally.",
                file=sys.stderr,
            )

    isolate = getattr(args, "isolate", False)
    quality_fast = getattr(args, "quality_fast", False)

    if isolate and args.mode not in ("timing", "load", "pr"):
        print(
            f"warning: --isolate is only supported for --mode timing; "
            f"ignoring for mode={args.mode!r}",
            file=sys.stderr,
        )
        isolate = False

    if isolate and args.mode == "load":
        print(
            "warning: --isolate is incompatible with --mode load (load mode is intentionally "
            "in-process to exercise the gate); ignoring --isolate",
            file=sys.stderr,
        )
        isolate = False

    if quality_fast and args.mode not in ("quality", "pr"):
        print(
            f"warning: --quality-fast is only meaningful for --mode quality or pr; "
            f"ignoring for mode={args.mode!r}",
            file=sys.stderr,
        )

    if args.mode == "quick":
        config = {"warmup": 0, "repeat": 1}
        iterations = run_quick_sync(cases)
    elif args.mode == "memory":
        config = {"warmup": 0, "repeat": 1, "tracemalloc": True}
        iterations = run_memory_sync(cases)
    elif args.mode == "accuracy":
        config = {"warmup": 0, "repeat": 1, "stages": ["estimate", "optimize"]}
        iterations = run_accuracy_sync(cases)
    elif args.mode == "quality":
        config = {
            "warmup": 0,
            "repeat": 1,
            "metrics": (
                ["ssim", "psnr"] if quality_fast else ["ssim", "psnr", "ssimulacra2", "butteraugli"]
            ),
            "quality_fast": quality_fast,
        }
        iterations = run_quality_sync(cases, fast=quality_fast)
    elif args.mode == "load":
        config = {
            "warmup": 0,
            "repeat": 1,
            "n_concurrent": args.n_concurrent,
            "semaphore_size": args.semaphore_size,
            "queue_depth": args.queue_depth,
            "memory_budget_mb": args.memory_budget_mb,
        }
        iterations = run_load_sync(
            cases,
            n_concurrent=args.n_concurrent,
            semaphore_size=args.semaphore_size,
            queue_depth=args.queue_depth,
            memory_budget_mb=args.memory_budget_mb,
        )
    elif args.mode == "pr":
        config = {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "stages": ["estimate", "optimize", "quality", "timing"],
            "quality_fast": quality_fast,
            "metrics": (
                ["ssim", "psnr"] if quality_fast else ["ssim", "psnr", "ssimulacra2", "butteraugli"]
            ),
        }
        iterations = run_pr_sync(
            cases,
            warmup=args.warmup,
            repeat=args.repeat,
            fast_quality=quality_fast,
        )
    else:  # timing
        config = {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "shuffle": not args.no_shuffle,
            "isolate": isolate,
        }
        iterations = run_timing_sync(
            cases,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            shuffle=not args.no_shuffle,
            isolate=isolate,
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
    allow_mode = getattr(args, "allow_mismatched_mode", False)
    allow_isolate = getattr(args, "allow_mismatched_isolate", False)
    allow_platform = getattr(args, "allow_mismatched_platform", False)

    try:
        result = compare(
            Path(args.baseline),
            Path(args.head),
            threshold_pct=args.threshold_pct,
            noise_floor_pct=args.noise_floor_pct,
            alpha=args.alpha,
            allow_mismatched_mode=allow_mode,
        )
    except ModeMismatchError as e:
        # Hard error: modes are incompatible and user did not opt in.
        print(f"compare: comparability error: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as e:
        # ValueError comes from load_run on schema_version mismatch;
        # KeyError catches missing required fields. Both map to
        # docstring exit code 2 ("schema error").
        print(f"compare: schema error: {e}", file=sys.stderr)
        return 2

    # Emit soft warnings for isolate / platform mismatches (only when not opted in).
    a = result.a_conditions
    b = result.b_conditions
    if a is not None and b is not None:
        if a.isolate != b.isolate and not allow_isolate:
            print(
                f"WARNING: isolate mismatch — baseline isolate={a.isolate} but "
                f"head isolate={b.isolate}. Isolated runs carry ~200-400ms/iter "
                f"subprocess overhead. Pass --allow-mismatched-isolate to suppress.",
                file=sys.stderr,
            )
        if a.platform != b.platform and not allow_platform:
            print(
                f"WARNING: platform mismatch — baseline platform={a.platform!r} but "
                f"head platform={b.platform!r}. Pillow/zlib version drift across OSes "
                f"may affect timings. Pass --allow-mismatched-platform to suppress.",
                file=sys.stderr,
            )

    if args.format == "markdown":
        print(render_compare_markdown(result))
    else:
        # JSON for piping into automation.
        import json
        from dataclasses import asdict

        conditions: dict[str, Any] = {}
        if a is not None and b is not None:
            conditions = {
                "baseline": asdict(a),
                "head": asdict(b),
            }

        print(
            json.dumps(
                {
                    "metadata": {
                        "baseline": str(Path(args.baseline).name),
                        "head": str(Path(args.head).name),
                        "conditions": conditions,
                    },
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
    p_run.add_argument(
        "--mode",
        choices=("quick", "timing", "memory", "accuracy", "quality", "load", "pr"),
        default="timing",
    )
    p_run.add_argument("--manifest", default="core")
    p_run.add_argument("--corpus", default=str(DEFAULT_CORPUS_ROOT))
    p_run.add_argument("--out", default="reports/bench.json")
    p_run.add_argument("--bucket", default=None)
    p_run.add_argument("--fmt", action="append", default=None)
    p_run.add_argument("--tag", default=None)
    p_run.add_argument(
        "--exclude-tag",
        default=None,
        dest="exclude_tag",
        metavar="TAG",
        help=(
            "skip entries whose tag list includes TAG "
            "(e.g. --exclude-tag fat_input keeps timing/quick/memory runs cheap)"
        ),
    )
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
    p_run.add_argument(
        "--isolate",
        action="store_true",
        help=(
            "run each iteration in a fresh Python subprocess "
            "(clean per-case RSS; +200-400ms/iter cold-start)"
        ),
    )
    p_run.add_argument("--annotate", action="append", default=None)
    p_run.add_argument(
        "--quality-fast",
        action="store_true",
        help="(quality mode) skip SSIMULACRA2 + butteraugli subprocess calls; "
        "compute only pure-numpy SSIM + PSNR (~50ms/case vs ~3.5s/case)",
    )
    # Load-mode flags (silently ignored for other modes).
    p_run.add_argument(
        "--n-concurrent",
        type=int,
        default=30,
        help="(load mode) number of concurrent optimize_image() calls per case",
    )
    p_run.add_argument(
        "--semaphore-size",
        type=int,
        default=8,
        help="(load mode) CompressionGate semaphore size (mirrors settings.compression_semaphore_size)",
    )
    p_run.add_argument(
        "--queue-depth",
        type=int,
        default=16,
        help="(load mode) CompressionGate max queue depth (mirrors settings.max_queue_depth)",
    )
    p_run.add_argument(
        "--memory-budget-mb",
        type=int,
        default=0,
        help="(load mode) CompressionGate memory budget in MB (0 = use settings default)",
    )
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="diff two runs with significance tests")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("head")
    p_cmp.add_argument("--threshold-pct", type=float, default=10.0)
    p_cmp.add_argument(
        "--noise-floor-pct",
        type=float,
        default=25.0,
        help=(
            "delta%% threshold used when either side has fewer than 3 iterations "
            "(noise-floor path). Default: 25.0"
        ),
    )
    p_cmp.add_argument("--alpha", type=float, default=0.05)
    p_cmp.add_argument("--format", choices=("markdown", "json"), default="markdown")
    p_cmp.add_argument(
        "--allow-mismatched-mode",
        action="store_true",
        dest="allow_mismatched_mode",
        help=(
            "skip the mode-mismatch error and compute diffs anyway. "
            "Use when you know the modes differ but want a rough trend comparison."
        ),
    )
    p_cmp.add_argument(
        "--allow-mismatched-isolate",
        action="store_true",
        dest="allow_mismatched_isolate",
        help=(
            "suppress the isolate-mismatch WARNING. "
            "Isolate adds ~200-400ms/iter subprocess overhead."
        ),
    )
    p_cmp.add_argument(
        "--allow-mismatched-platform",
        action="store_true",
        dest="allow_mismatched_platform",
        help=(
            "suppress the platform-mismatch WARNING. "
            "Pillow/zlib version drift across OSes may affect timings."
        ),
    )
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

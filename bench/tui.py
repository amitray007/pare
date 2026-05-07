"""Interactive TUI wizard for bench commands.

Usage::

    python -m bench.tui

Walks team members through the four common bench flows without requiring
them to memorise flags.  Each step prints the equivalent CLI command so
users learn the underlying tools.

Requires: questionary>=2.0
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import questionary

# ---------------------------------------------------------------------------
# Bootstrap — resolve the Python executable to use for all subprocesses
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASELINE = "reports/baseline.core.json"
DEFAULT_HEAD = "reports/head.json"
DEFAULT_DASH_OUT = "/tmp/dash"
SMOKE_OUT = "/tmp/smoke.json"

MODES = [
    questionary.Choice("pr     — combined timing + accuracy + quality (recommended)", "pr"),
    questionary.Choice("quick  — fastest, sanity smoke", "quick"),
    questionary.Choice("timing — timing percentiles only (multi-iter)", "timing"),
    questionary.Choice("memory — peak RSS measurement", "memory"),
    questionary.Choice("accuracy — estimator vs actual", "accuracy"),
    questionary.Choice("quality — SSIM/PSNR/SSIMULACRA2 only", "quality"),
    questionary.Choice("load   — concurrency stress test", "load"),
]

MANIFESTS = [
    questionary.Choice("core (synthetic, ~55 cases)", "core"),
    questionary.Choice("full (real-world, ~97 cases — needs `bench.corpus fetch` first)", "full"),
]

BUCKETS = ["all", "tiny", "small", "medium", "large", "xlarge"]

# ---------------------------------------------------------------------------
# Pure argument-builder functions (testable without interactivity)
# ---------------------------------------------------------------------------


def build_run_args(
    *,
    mode: str,
    manifest: str,
    fmt_filter: str,
    bucket: str,
    repeat: int,
    warmup: int,
    out: str,
    annotate: str = "",
) -> list[str]:
    """Return the argv list for ``python -m bench.run`` (without the executable).

    Parameters
    ----------
    mode:
        One of the bench mode strings (``pr``, ``quick``, etc.).
    manifest:
        ``core`` or ``full``.
    fmt_filter:
        Comma-separated format names, e.g. ``"jpeg,png"``.  Empty string = no filter.
    bucket:
        One of the BUCKETS list.  ``"all"`` means no filter.
    repeat:
        ``--repeat`` value (number of iterations).
    warmup:
        ``--warmup`` value.
    out:
        Output JSON path.
    annotate:
        Raw ``KEY=VAL`` annotation string.  Empty string = no annotation.
    """
    args = ["-m", "bench.run", "--mode", mode, "--manifest", manifest]
    if fmt_filter.strip():
        for fmt in fmt_filter.replace(" ", "").split(","):
            if fmt:
                args += ["--fmt", fmt]
    if bucket and bucket != "all":
        args += ["--bucket", bucket]
    if mode in ("timing", "pr"):
        args += ["--repeat", str(repeat), "--warmup", str(warmup)]
    args += ["--out", out]
    if annotate.strip():
        args += ["--annotate", annotate.strip()]
    return args


def build_compare_args(
    *,
    baseline: str,
    head: str,
    threshold_pct: float,
    noise_floor_pct: float,
    fmt: str,
) -> list[str]:
    """Return the argv list for ``python -m bench.compare``.

    Parameters
    ----------
    baseline:
        Path to baseline JSON.
    head:
        Path to head JSON.
    threshold_pct:
        ``--threshold-pct`` value.
    noise_floor_pct:
        ``--noise-floor-pct`` value.
    fmt:
        Output format: ``"markdown"`` or ``"json"``.
    """
    return [
        "-m",
        "bench.compare",
        baseline,
        head,
        "--threshold-pct",
        str(threshold_pct),
        "--noise-floor-pct",
        str(noise_floor_pct),
        "--format",
        fmt,
    ]


def build_dashboard_args(
    *,
    baseline: str,
    out_dir: str,
    with_samples: bool,
) -> list[str]:
    """Return the argv list for ``python -m bench.dashboard.build``."""
    args = ["-m", "bench.dashboard.build", "--baseline", baseline, "--out-dir", out_dir]
    if not with_samples:
        args.append("--no-with-samples")
    return args


def find_free_port(start: int = 8765, attempts: int = 20) -> int:
    """Return the first free TCP port starting from *start*.

    Raises ``OSError`` if no free port is found within *attempts*.
    """
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start}–{start + attempts - 1}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corpus_is_built(manifest: str) -> bool:
    """Return True if the corpus for *manifest* appears to be built on disk."""
    corpus_root = _PROJECT_ROOT / "tests" / "corpus"
    if not corpus_root.exists():
        return False
    # Any non-hidden file means at least one image is present.
    return any(corpus_root.iterdir())


def _print_cmd(args: list[str]) -> None:
    """Print the command that is about to be run (readable form)."""
    display = [PYTHON] + args
    print(f"\n  {' '.join(display)}\n")


def _run(args: list[str]) -> int:
    """Run ``PYTHON *args`` streaming output to the terminal.  Returns exit code."""
    _print_cmd(args)
    result = subprocess.run([PYTHON] + args, check=False)
    return result.returncode


def _ask_int(message: str, default: int, *, min_val: int = 0) -> int:
    """Ask for a positive integer with a default."""

    def validate(raw: str) -> bool | str:
        try:
            v = int(raw)
        except ValueError:
            return "Please enter a whole number."
        if v < min_val:
            return f"Must be >= {min_val}."
        return True

    raw = questionary.text(
        message,
        default=str(default),
        validate=validate,
    ).ask()
    if raw is None:
        raise KeyboardInterrupt
    return int(raw)


def _ask_path(message: str, default: str) -> str:
    raw = questionary.text(message, default=default).ask()
    if raw is None:
        raise KeyboardInterrupt
    return raw.strip() or default


def _done_what_next(options: list[questionary.Choice | str]) -> str | None:
    print()
    answer = questionary.select("What next?", choices=options + ["Quit"]).ask()
    return answer


# ---------------------------------------------------------------------------
# Flow 1 — Run a benchmark
# ---------------------------------------------------------------------------


def flow_run() -> str | None:
    """Interactive flow for running a benchmark.  Returns a what-next token."""
    mode = questionary.select("What mode?", choices=MODES).ask()
    if mode is None:
        return None

    manifest_answer = questionary.select("Which manifest?", choices=MANIFESTS).ask()
    if manifest_answer is None:
        return None
    manifest: str = manifest_answer

    # Corpus check
    if not _corpus_is_built(manifest):
        build_now = questionary.confirm(
            f"The corpus for '{manifest}' doesn't look built. Build it now?",
            default=True,
        ).ask()
        if build_now is None:
            return None
        if build_now:
            rc = _run(["-m", "bench.corpus", "build", "--manifest", manifest])
            if rc != 0:
                print(f"\nCorpus build exited with code {rc}. Continuing anyway.")

    fmt_raw = questionary.text(
        "Filter to specific format(s)? (comma-separated, blank for all)",
        default="",
    ).ask()
    if fmt_raw is None:
        return None

    bucket = questionary.select("Filter to specific bucket?", choices=BUCKETS).ask()
    if bucket is None:
        return None

    repeat = _ask_int("Iterations (--repeat)", default=3, min_val=1)
    warmup = _ask_int("Warmup (--warmup)", default=1, min_val=0)

    out = _ask_path("Output path", DEFAULT_HEAD)

    run_args = build_run_args(
        mode=mode,
        manifest=manifest,
        fmt_filter=fmt_raw or "",
        bucket=bucket,
        repeat=repeat,
        warmup=warmup,
        out=out,
    )

    print("\nAbout to run:")
    _print_cmd(run_args)

    proceed = questionary.confirm("Proceed?", default=True).ask()
    if not proceed:
        return None

    rc = subprocess.run([PYTHON] + run_args, check=False).returncode
    if rc != 0:
        print(f"\nBenchmark exited with code {rc}.")

    baseline = str(_PROJECT_ROOT / DEFAULT_BASELINE)
    choices = []
    if Path(baseline).exists():
        choices.append(questionary.Choice(f"Compare against {DEFAULT_BASELINE}", "compare"))
    choices += [
        questionary.Choice("Render markdown report", "report"),
        questionary.Choice("Build dashboard preview", "dashboard"),
        questionary.Choice("Run another benchmark", "run"),
    ]
    return _done_what_next(choices)


# ---------------------------------------------------------------------------
# Flow 2 — Compare two runs
# ---------------------------------------------------------------------------


def flow_compare(
    *, prefill_baseline: str = DEFAULT_BASELINE, prefill_head: str = DEFAULT_HEAD
) -> str | None:
    """Interactive flow for comparing two bench runs."""
    baseline = _ask_path("Baseline file", prefill_baseline)
    head = _ask_path("Head file", prefill_head)

    # Graceful skip if one of them is missing.
    missing = [p for p in (baseline, head) if not Path(p).exists()]
    if missing:
        for p in missing:
            print(f"  File not found: {p}")
        skip = questionary.confirm("Skip comparison (files missing)?", default=True).ask()
        if skip or skip is None:
            return None

    threshold = _ask_int("Threshold pct (timing significance)", default=10, min_val=1)
    noise_floor = _ask_int("Noise-floor pct (low-iter fallback)", default=25, min_val=1)

    fmt = questionary.select(
        "Output format?",
        choices=[
            questionary.Choice("markdown (printed to terminal)", "markdown"),
            questionary.Choice("json", "json"),
        ],
    ).ask()
    if fmt is None:
        return None

    cmp_args = build_compare_args(
        baseline=baseline,
        head=head,
        threshold_pct=threshold,
        noise_floor_pct=noise_floor,
        fmt=fmt,
    )
    _run(cmp_args)

    choices = [
        questionary.Choice("Build dashboard preview from head", "dashboard"),
        questionary.Choice("Run another comparison", "compare"),
    ]
    return _done_what_next(choices)


# ---------------------------------------------------------------------------
# Flow 3 — Build dashboard preview
# ---------------------------------------------------------------------------


def flow_dashboard(*, prefill_baseline: str = DEFAULT_BASELINE) -> str | None:
    """Interactive flow for building the dashboard."""
    baseline = _ask_path("Baseline file (the JSON to render)", prefill_baseline)
    out_dir = _ask_path("Output directory", DEFAULT_DASH_OUT)
    with_samples = questionary.confirm(
        "Generate quality samples (with thumbnails)?", default=True
    ).ask()
    if with_samples is None:
        return None

    dash_args = build_dashboard_args(
        baseline=baseline,
        out_dir=out_dir,
        with_samples=with_samples,
    )
    rc = _run(dash_args)

    index_html = Path(out_dir) / "index.html"
    if rc == 0 and index_html.exists():
        size_kb = index_html.stat().st_size // 1024
        print(f"\nBuilt {index_html} ({size_kb} KB)")
    else:
        print(f"\nDashboard build exited with code {rc}.")

    serve = questionary.confirm("Start a local HTTP server to preview?", default=True).ask()
    if serve:
        try:
            port = find_free_port()
        except OSError as exc:
            print(f"Could not find a free port: {exc}")
            return None

        print(f"\n  python -m http.server {port} --directory {out_dir}")
        print(f"  -> http://localhost:{port}/")
        print("  Press Ctrl+C to stop the server and return to the menu.\n")
        server = subprocess.Popen(
            [PYTHON, "-m", "http.server", str(port), "--directory", out_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            server.wait()
        except KeyboardInterrupt:
            server.terminate()
            server.wait()
            print("\nServer stopped.")

    choices = [
        questionary.Choice("Run a benchmark", "run"),
        questionary.Choice("Compare two runs", "compare"),
    ]
    return _done_what_next(choices)


# ---------------------------------------------------------------------------
# Flow 4 — Quick smoke
# ---------------------------------------------------------------------------


def flow_smoke() -> str | None:
    """One-shot quick-smoke, no questions asked."""
    smoke_args = build_run_args(
        mode="pr",
        manifest="core",
        fmt_filter="jpeg",
        bucket="small",
        repeat=1,
        warmup=0,
        out=SMOKE_OUT,
    )
    print(f"\nRunning quick smoke: pr mode, jpeg + small, 1 iter, no warmup, output {SMOKE_OUT}")
    rc = _run(smoke_args)
    if rc == 0:
        out_path = Path(SMOKE_OUT)
        if out_path.exists():
            print(f"\nDone. {SMOKE_OUT} ({out_path.stat().st_size // 1024} KB)")
        else:
            print(f"\nDone (exit 0), but {SMOKE_OUT} wasn't created.")
    else:
        print(f"\nSmoke exited with code {rc}.")
        return None

    # Auto-compare against baseline if it exists.
    baseline = _PROJECT_ROOT / DEFAULT_BASELINE
    if baseline.exists():
        auto = questionary.confirm(f"Auto-compare against {DEFAULT_BASELINE}?", default=True).ask()
        if auto:
            cmp_args = build_compare_args(
                baseline=str(baseline),
                head=SMOKE_OUT,
                threshold_pct=10,
                noise_floor_pct=25,
                fmt="markdown",
            )
            _run(cmp_args)

    choices = [
        questionary.Choice("Run a full benchmark", "run"),
        questionary.Choice("Compare two runs", "compare"),
    ]
    return _done_what_next(choices)


# ---------------------------------------------------------------------------
# Top-level menu loop
# ---------------------------------------------------------------------------

TOP_MENU = [
    questionary.Choice("Run a benchmark", "run"),
    questionary.Choice("Compare two runs", "compare"),
    questionary.Choice("Build dashboard preview", "dashboard"),
    questionary.Choice("Quick smoke (fastest path: jpeg + small + 1 iter)", "smoke"),
    questionary.Choice("Quit", "quit"),
]


def main() -> None:
    """Entry point for ``python -m bench.tui``."""
    print()
    next_action: str | None = None

    while True:
        if next_action is None:
            choice = questionary.select("What do you want to do?", choices=TOP_MENU).ask()
            if choice is None:
                # Ctrl+C at the main menu
                break
        else:
            choice = next_action
            next_action = None

        if choice == "quit" or choice is None:
            break
        elif choice == "run":
            next_action = flow_run()
        elif choice == "compare":
            next_action = flow_compare()
        elif choice == "dashboard":
            next_action = flow_dashboard()
        elif choice == "smoke":
            next_action = flow_smoke()
        elif choice == "report":
            # Reached via what-next from a run.
            head = _ask_path("Run JSON to render", DEFAULT_HEAD)
            _run(["-m", "bench.run", "report", head, "--format", "markdown"])
            next_action = None


if __name__ == "__main__":
    main()

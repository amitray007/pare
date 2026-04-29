"""Top-level dispatcher: `python -m bench` prints subcommand help."""

import sys

USAGE = """\
Usage: python -m bench <subcommand> [options]

Subcommands:
  corpus    Build, verify, and list corpus manifests
  run       Run benchmarks (modes: quick, timing, memory)
  compare   Diff two benchmark runs with statistical significance
  report    Render a run as markdown / JSON

Examples:
  python -m bench.corpus build --manifest core
  python -m bench.run --mode timing --fmt jpeg
  python -m bench.compare reports/a.json reports/b.json
"""


def main() -> int:
    sys.stderr.write(USAGE)
    return 1 if len(sys.argv) > 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())

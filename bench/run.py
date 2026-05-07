"""Top-level entry point for `python -m bench.run`.

Thin shim around `bench.runner.cli.main`. Importing this module starts
the CLI so `python -m bench.run --mode timing` works without a
`bench.runner` prefix.
"""

from bench.runner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

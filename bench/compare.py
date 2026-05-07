"""`python -m bench.compare A B` shortcut. Delegates to bench.runner.cli."""

import sys

from bench.runner.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["compare", *sys.argv[1:]]))

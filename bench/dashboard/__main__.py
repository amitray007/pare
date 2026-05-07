"""Entry point: ``python -m bench.dashboard.build`` delegates here."""

import sys

from bench.dashboard.build import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

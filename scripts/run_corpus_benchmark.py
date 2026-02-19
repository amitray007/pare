"""Run the full corpus benchmark across all formats and presets.

Cross-platform launcher that:
  1. Validates the corpus exists (with download hint if missing)
  2. Runs benchmarks for all formats and presets
  3. Saves HTML + JSON reports to reports/
  4. Opens the HTML report in the default browser

Usage:
    python scripts/run_corpus_benchmark.py                  # Full run
    python scripts/run_corpus_benchmark.py --fmt jpeg       # Single format
    python scripts/run_corpus_benchmark.py --preset high    # Single preset
    python scripts/run_corpus_benchmark.py --no-open        # Don't open browser
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "tests" / "corpus"
REPORTS_DIR = ROOT / "reports"


def check_corpus():
    if not CORPUS_DIR.is_dir():
        print(f"Corpus not found at {CORPUS_DIR}")
        print("Download it first:")
        print(f"  python scripts/download_unsplash_corpus.py --key YOUR_KEY")
        print(f"  python scripts/convert_corpus_formats.py")
        sys.exit(1)

    image_exts = {".jpg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".avif", ".heic", ".jxl"}
    count = sum(1 for f in CORPUS_DIR.rglob("*") if f.suffix.lower() in image_exts)
    if count == 0:
        print(f"No image files found in {CORPUS_DIR}")
        sys.exit(1)

    return count


def find_latest_report() -> Path | None:
    if not REPORTS_DIR.exists():
        return None
    htmls = sorted(REPORTS_DIR.glob("benchmark-*.html"), reverse=True)
    return htmls[0] if htmls else None


def main():
    parser = argparse.ArgumentParser(description="Run Pare corpus benchmarks")
    parser.add_argument("--fmt", help="Filter by format (jpeg, png, webp, gif, bmp, tiff, avif, heic, jxl)")
    parser.add_argument("--preset", help="Run only this preset (high, medium, low)")
    parser.add_argument("--no-open", action="store_true", help="Don't open HTML report in browser")
    parser.add_argument("--json", action="store_true", help="Also print JSON to stdout")
    args = parser.parse_args()

    count = check_corpus()
    print(f"Corpus: {count} images in {CORPUS_DIR}")
    print()

    # Build benchmark command
    cmd = [sys.executable, "-m", "benchmarks.run", "--corpus", str(CORPUS_DIR)]
    if args.fmt:
        cmd += ["--fmt", args.fmt]
    if args.preset:
        cmd += ["--preset", args.preset]
    if args.json:
        cmd += ["--json"]

    # Run benchmark
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\nBenchmark failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Open report
    if not args.no_open:
        report = find_latest_report()
        if report:
            print(f"\nOpening report: {report}")
            webbrowser.open(report.as_uri())


if __name__ == "__main__":
    main()

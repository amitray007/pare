# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Benchmark suite that measures optimization effectiveness, speed, and estimation accuracy across all formats and quality presets.

## Running Benchmarks

```bash
python -m benchmarks.run                              # All formats, all presets
python -m benchmarks.run --fmt png                    # Single format
python -m benchmarks.run --preset high --fmt bmp      # Single preset + format
python -m benchmarks.run --compare                    # Delta between last two runs
python -m benchmarks.run --no-save                    # Don't persist reports
python -m benchmarks.run --json                       # JSON to stdout
```

Reports are saved to `reports/` as timestamped HTML + JSON files.

## Architecture

- **`constants.py`**: Quality presets (HIGH q=40, MEDIUM q=60, LOW q=80), size definitions, format-specific quality levels. Single source of truth for preset configs.
- **`cases.py`**: `BenchmarkCase` definitions — pairs content generators with formats/sizes. `build_all_cases()` generates the full matrix.
- **`generators.py`**: Deterministic image generators (seeded RNG): `photo_like`, `screenshot_like`, `graphic_like`, `gradient`, `solid`, `transparent_png`, SVG variants. Used by cases.py.
- **`runner.py`**: Executes optimization + estimation concurrently per case. Pre-computes header analysis once per image (shared across presets). Semaphore-bounded parallelism.
- **`report.py`**: Console table, HTML dashboard, and JSON export. The console report includes format summaries, per-case details, and estimation accuracy tables.
- **`run.py`**: CLI entry point with argparse. Also handles `--compare` mode (delta between two JSON reports).

## Key Metrics

- **Preset differentiation**: HIGH should show higher reduction than MEDIUM, which should show higher than LOW
- **Estimation accuracy**: "Avg Err" column in the ESTIMATION ACCURACY section. Target: <15%
- **Method selection**: Verify the right methods appear per preset (e.g., lossy methods only in HIGH/MEDIUM)

## Adding Benchmark Cases

Add cases in `cases.py` by creating `BenchmarkCase` instances with a generator from `generators.py`. Each case needs: `name`, `data` (bytes), `fmt` (format string), `category` (size label), `content` (content type label).

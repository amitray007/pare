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

# Corpus benchmarks (real-world images)
python -m benchmarks.run --corpus tests/corpus                         # All groups
python -m benchmarks.run --corpus tests/corpus --group high_res        # Single group
python -m benchmarks.run --corpus tests/corpus --group high_res standard --fmt jpeg

# Dashboard server
python -m benchmarks.server                           # http://localhost:8081
```

Reports are saved to `reports/` as timestamped HTML + JSON files.

## Architecture

- **`constants.py`**: Quality presets (HIGH q=40, MEDIUM q=60, LOW q=80), size definitions, corpus group definitions (`CorpusGroup`, `CORPUS_GROUPS`). Single source of truth for preset and group configs.
- **`cases.py`**: `BenchmarkCase` definitions — pairs content generators with formats/sizes. `build_all_cases()` generates the full matrix. `BenchmarkCase.group` field holds corpus group assignment.
- **`corpus.py`**: Group-aware corpus loader. Reads `groups.json` manifest for group assignment; falls back to dimension-based classification. Provides `load_corpus_cases()` and `scan_corpus_by_group()`.
- **`generators.py`**: Deterministic image generators (seeded RNG): `photo_like`, `screenshot_like`, `graphic_like`, `gradient`, `solid`, `transparent_png`, SVG variants. Also encodes to all raster formats (PNG, JPEG, WebP, GIF, BMP, TIFF, AVIF, HEIC, JXL). Used by cases.py.
- **`runner.py`**: Executes optimization + estimation concurrently per case. Uses sample-based estimation via `estimation.estimator.estimate()`. Semaphore-bounded parallelism.
- **`report.py`**: Console table, HTML dashboard, and JSON export. The console report includes format summaries, per-case details, and estimation accuracy tables.
- **`run.py`**: CLI entry point with argparse. Supports `--group` filter for corpus group selection. Also handles `--compare` mode (delta between two JSON reports).
- **`server.py`**: Dashboard server. `/api/corpus` returns group info, `/api/run` accepts `groups` filter, SSE stream includes `group` field per result.

## Corpus Groups

The benchmark corpus is organized into 4 technical groups in `tests/corpus/groups.json`:

| Group | Directory | Content | Dimensions |
|-------|-----------|---------|------------|
| `high_res` | `tests/corpus/high_res/` | Landscape, architecture, texture | 2400px, >500KB |
| `standard` | `tests/corpus/standard/` | Portrait, food, macro | 1200px, 100-500KB |
| `compact` | `tests/corpus/compact/` | Abstract, monochrome, colorful | 400px, <100KB |
| `deep_color` | `tests/corpus/deep_color/` | Native AVIF (10/12-bit) | Varies |

Each group (except deep_color) has 3 photos in 9 formats (JPEG, PNG, AVIF, WebP, BMP, TIFF, GIF, HEIC, JXL) = 27 files per group. Deep_color has 8 native AVIF samples.

## Key Metrics

- **Preset differentiation**: HIGH should show higher reduction than MEDIUM, which should show higher than LOW
- **Estimation accuracy**: "Avg Err" column in the ESTIMATION ACCURACY section. Target: <15%
- **Method selection**: Verify the right methods appear per preset (e.g., lossy methods only in HIGH/MEDIUM)
- **Per-group breakdown**: Dashboard shows per-group reduction averages in overview cards

## Adding Benchmark Cases

Add cases in `cases.py` by creating `BenchmarkCase` instances with a generator from `generators.py`. Each case needs: `name`, `data` (bytes), `fmt` (format string), `category` (size label), `content` (content type label), `group` (corpus group, optional).

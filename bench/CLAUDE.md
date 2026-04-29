# bench/

Pare benchmarking + corpus toolkit. Replaces the legacy `benchmarks/` and `scripts/{download,convert}_corpus*.py`.

## Why this exists separately from `benchmarks/`

The legacy `benchmarks/` measures `time.process_time()` (parent only) and `tracemalloc` (Python heap only). MozJPEG, pngquant, oxipng, cjxl etc. run as subprocesses and account for 80–95 % of CPU work, so the legacy CPU and memory numbers under-report by an order of magnitude. `bench/` uses `RUSAGE_CHILDREN` deltas and `ru_maxrss` of children to capture honest totals.

## Subpackages

- **`bench.corpus`** — deterministic, manifest-driven corpus builder. Synthesizers produce byte-identical raw pixel data from `(kind, seed, dims)`. Manifests pin pixel-level SHA-256 (decoded `Image.tobytes()`), not encoded SHA — encoded bytes drift across libjpeg-turbo SIMD paths.
- **`bench.runner`** — subprocess-aware benchmark runner. Modes: `quick` (1 iter, smoke), `timing` (5 iter + warmup, p50/p95/p99 + MAD, `--isolate`), `memory` (1 iter, peak RSS headline). `load` mode deferred to v1.

## Determinism contract

- Pixel-level SHA-256 of raw `Image.tobytes()` is canonical.
- Encoded SHA may be recorded as `encoded_sha256.<platform>` for diagnostics, but is never blocking.
- `random.Random(seed)` instances only — never mutate the global PRNG.
- Fonts vendored at `bench/corpus/fonts/` (Pillow's default font is build-dependent).

## Common commands

```bash
# Build the canonical 30-case corpus
python -m bench.corpus build --manifest core

# Verify the on-disk corpus matches the manifest pixel-hashes
python -m bench.corpus verify --manifest core

# Quick smoke (1 iter, all formats, ~1 min)
python -m bench.run --mode quick

# Honest latency (5 iters, isolated, p50/p95/p99)
python -m bench.run --mode timing --manifest core --out reports/timing.json

# Peak RSS truth (1 iter, isolated)
python -m bench.run --mode memory --out reports/memory.json

# Diff two runs with Welch's t-test
python -m bench.compare reports/baseline.json reports/head.json --threshold-pct 10
```

## Code style

Same as repo root: Black 100 cols, Ruff E/F/W/I, Python 3.12, pytest-asyncio.

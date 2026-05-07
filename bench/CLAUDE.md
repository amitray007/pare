# bench/

Pare benchmarking + corpus toolkit. Replaces the legacy `benchmarks/` and `scripts/{download,convert}_corpus*.py`.

## Why this exists separately from `benchmarks/`

The legacy `benchmarks/` measures `time.process_time()` (parent only) and `tracemalloc` (Python heap only). MozJPEG, pngquant, oxipng, cjxl etc. run as subprocesses and account for 80–95 % of CPU work, so the legacy CPU and memory numbers under-report by an order of magnitude. `bench/` uses `RUSAGE_CHILDREN` deltas and `ru_maxrss` of children to capture honest totals.

## Two-tier corpus

The corpus has two named manifests that can be benchmarked independently or together:

**`core` — synthesized** (`bench/corpus/manifests/core.json`, manifest_version=1, **103 entries**):
- Images generated deterministically from `(kind, seed, dims)` via `bench/corpus/synthesis/`.
- No network required; reproducible anywhere from source code alone.
- Pixel-level SHA-256 is pinned in the manifest (`expected_pixel_sha256`).
- Covers all 12 formats across small/medium/large/xlarge buckets with ≥5 entries per content_kind; two format-limit exceptions noted in the manifest comments (AVIF graphic_geometric@large, WebP path_thin_gradient@large).

**`full` — fetched + hash-pinned** (`bench/corpus/manifests/full.json`, manifest_version=2, **111 entries**):
- Real-world images downloaded from declared URLs (Kodak Lossless True Color Image Suite + Wikimedia Commons non-photo content).
- Each entry has a `source` field with `url`, `sha256`, `license`, and `attribution`.
- Fetched bytes are cached under `bench/corpus/cache/<sha256[:2]>/<sha256>/<basename>` — not committed; add to `.gitignore`.
- Pixel-level SHA-256 is still pinned per entry — the pixel hash catches CDN re-encodes.
- Use `python -m bench.corpus fetch --manifest full` to pre-warm the cache before building.
- Six non-photo `fetched_*` content_kinds cover categories where synthetic data is misleading: `fetched_text_screenshot`, `fetched_graphic_palette`, `fetched_graphic_geometric`, `fetched_transparent_overlay`, `fetched_animated_redraw`, `fetched_path_flat_text`. All sourced from Wikimedia Commons (CC-compatible / public domain). See `bench/corpus/synthesis/fetched.py` for the raise-on-call stub pattern.

### v1 → v2 schema change

`MANIFEST_VERSION` was bumped from `1` to `2` in `bench/corpus/manifest.py`. `Manifest.from_json()` accepts both `{1, 2}`. v1 entries have no `source` field (`source=None` after loading). No structural changes to existing synthesized entries are required.

A new `SourceSpec` dataclass was added with fields: `url`, `sha256`, `license`, `attribution`, `notes` (optional). Two stubs are registered: `fetched_photo` (raster) and `fetched_vector` (vector) — both raise `RuntimeError` if synthesis is accidentally invoked; the builder routes these through `fetch()` instead.

A new `expected_byte_sha256` field (`dict[str, str] | None`) was added to `ManifestEntry` for vector entries. Raster entries leave this `None` and use `expected_pixel_sha256` as before. Vector entries leave `expected_pixel_sha256` as `None` and store `{"source": "<sha256>"}` in `expected_byte_sha256`.

## Subpackages

- **`bench.corpus`** — manifest-driven corpus builder. Synthesizers produce deterministic pixel data from seeds; fetchers download hash-pinned real-world images. Raster manifests pin pixel-level SHA-256 (decoded `Image.tobytes()`); vector manifests (SVG/SVGZ) pin byte-level SHA-256 of the source file. Encoded raster bytes drift across libjpeg-turbo SIMD paths and are never the canonical hash; vector encoded bytes are deterministic (mtime=0 gzip) and can be hashed but the source SHA is the contract. SVG synthesis (`bench/corpus/synthesis/svg.py`) has two content kinds: `vector_geometric` (rects/circles/lines on a coloured background, exercises scour path optimisation) and `vector_with_script` (same shapes plus a `<script>` block and `onclick` handler on the root element, exercises SvgOptimizer's sanitisation pipeline). SVG byte size scales with shape count, not with pixel dimensions — practical max bucket is ~medium without absurd shape counts.
- **`bench.corpus.fetchers`** — HTTP fetcher (`bench/corpus/fetchers/http.py`). Downloads to a content-addressed local cache (`bench/corpus/cache/`); verifies SHA-256 before returning the path. Exceptions: `FetchError`, `FetchIntegrityError`, `FetchHTTPError`, `FetchTooLargeError`. Default cache root overridable via `--cache PATH`.
- **`bench.runner`** — subprocess-aware benchmark runner. Modes: `quick` (1 iter, smoke), `timing` (5 iter + warmup, p50/p95/p99 + MAD, `--isolate`), `memory` (1 iter, peak RSS headline), `load` (concurrent request storm, gate testing), `accuracy` (estimate vs optimize size comparison). `load` mode per-case output includes `n_503_queue` (queue-depth cap hit) and `n_503_memory` (memory-budget cap hit) separately — `n_503` is their sum and is kept for backwards compatibility. Use `--tag fat_input` to restrict to the gate-exercising xlarge entries.

## Determinism contract

- **Raster entries**: Pixel-level SHA-256 of raw `Image.tobytes()` is canonical (field `expected_pixel_sha256`). Encoded SHA may be recorded as `encoded_sha256.<platform>` for diagnostics, but is never blocking.
- **Vector entries** (SVG/SVGZ): Byte-level SHA-256 of the raw source bytes is canonical (field `expected_byte_sha256["source"]`). SVG sources are XML; no pixel data exists. Encoded bytes are deterministic across platforms (no SIMD variance), so a flat `{format: sha256}` mapping suffices.
- **Deep-color entries** (10/12-bit): Pixel-level SHA-256 is computed from the raw `numpy.uint16` array bytes (dtype + shape are baked into the digest, so a 10-bit and a 16-bit array of the same logical pixels never collide). See `manifest.py:pixel_sha256()`.
- `random.Random(seed)` instances only — never mutate the global PRNG.
- Fonts vendored at `bench/corpus/fonts/` (Pillow's default font is build-dependent).
- Fetched raster entries: pixel hash computed from decoded source image (same `pixel_sha256()` function). Source URL SHA-256 guards against corrupt downloads; pixel SHA-256 guards against CDN re-encoding.
- Fetched vector entries (`fetched_vector` content_kind): source SHA-256 guards against corrupt downloads. The builder writes bytes directly to disk — no `Image.open()` is called.

## Deep-color encoding (10/12-bit)

`bench/corpus/synthesis/deep_color.py` produces `numpy.uint16` arrays with values in the raw bit-depth range: `[0, 1023]` for 10-bit, `[0, 4095]` for 12-bit. Bit depth is auto-detected from `max(array)` in `conversion._detect_bit_depth()`.

**JXL** — natively supported via `jxlpy.JXLPyEncoder` typed buffers. Accepts uint16 pixel data directly and encodes with the correct bit depth in the output container. The decoder (`jxlpy.JXLPyDecoder`) round-trips `bits_per_sample` in the info dict.

**HEIC/AVIF** — supported via `pillow_heif.from_bytes()` typed-buffer API (modes like `RGB;10`). Requires pillow_heif ≥ 0.22. Both succeed on pillow_heif 0.22.0 (tested). Raises `FormatNotSupportedError` with a descriptive message if the typed-buffer path fails.

**All other formats** (PNG, JPEG, WebP, GIF, BMP, TIFF, APNG) — ndarray content raises `FormatNotSupportedError` because 8-bit codecs cannot represent 10/12-bit pixel values without quantizing. This is intentional and correct.

**Manifest `bit_depth` field**: `ManifestEntry` has an optional `bit_depth: int | None` field (default `None` ≡ 8-bit). Deep-color entries set this to `10` or `12`. The field is omitted from JSON when `None`; present when set.

**Deferred**: The production estimator's BPP curve fits in `estimation/estimator.py` assume 8-bit input; deep-color accuracy may be off until estimator updates land in a follow-up PR.

## Fat-input corpus tier

The `core` manifest contains a `fat_input` tag group: 4 entries (TIFF, PNG, BMP, AVIF) with
encoded sizes in the 20–26 MB range — close to the 32 MB production upload limit.  These entries
exist to exercise the `CompressionGate` memory-budget gate under realistic worst-case memory
pressure; during a normal timing or memory run, the gate never engages because the rest of the
corpus tops out at ~8 MB.

**Sizes and memory estimates (encoded_bytes × MEMORY_MULTIPLIER):**

| entry | format | size | multiplier | est. memory |
|-------|--------|------|-----------|-------------|
| `fat_tiff_perlin_xlarge` | tiff | 24.3 MB | 6 | ~146 MB |
| `fat_png_noise_xlarge`   | png  | 22.0 MB | 5 | ~110 MB |
| `fat_bmp_noise_xlarge`   | bmp  | 22.0 MB | 3 |  ~66 MB |
| `fat_avif_noise_xlarge`  | avif | 23.5 MB | 4 |  ~94 MB |

**Tag-exclusion design decision:** A `--exclude-tag TAG` flag was added to `bench.run` (see
`bench/runner/case.py:load_cases` and `bench/runner/cli.py`).  This is **additive** — it has no
effect on any existing invocation unless someone passes the flag.  The alternative of introducing a
`huge` bucket and baking bucket exclusion into mode defaults was rejected because it would require
changing the manifest schema *and* hard-coding mode knowledge into `load_cases`, which is harder to
reverse.

**Usage:**

```bash
# Build fat-input corpus files (takes ~10s for AVIF)
python -m bench.corpus build --manifest core --tag fat_input

# Run the load gate test (budget=256 MB forces memory rejections for TIFF)
python -m bench.run --mode load --manifest core --tag fat_input \
  --semaphore-size 10 --queue-depth 20 --n-concurrent 30 --memory-budget-mb 256 \
  --out /tmp/load-fat-test.json

# Exclude fat_input from normal timing/quick/memory runs to keep them cheap
python -m bench.run --mode timing --exclude-tag fat_input
python -m bench.run --mode quick  --exclude-tag fat_input
```

## Estimator path attribution

`EstimateResponse` includes an optional `path` field that identifies which code path produced the estimate. Three values are possible:

| `path` | when used | error profile |
|--------|-----------|---------------|
| `exact` | images <150K pixels, SVG, animated | 0% by construction — full optimizer ran |
| `direct_encode_sample` | JPEG, HEIC, AVIF, JXL, WebP, PNG | BPP extrapolation drift; source of most outliers |
| `generic_fallback_sample` | GIF, BMP, TIFF | Sample compressed with actual optimizer; extrapolated |

**Why this matters**: before `path` attribution, per-format error averages blended exact-mode rows (always 0%) with sample-mode rows (where real drift lives), masking the signal. The PNG +197% bug and AVIF -83% bug were only isolatable once exact rows were separated from `direct_encode_sample` rows.

**Usage in accuracy analysis**:

```bash
# Run accuracy mode and pipe through analyzer
python -m bench.run --mode accuracy --manifest full --out reports/accuracy.json
python reports/full-bench/analyze_accuracy.py reports/accuracy.json
# Look at "=== Estimator error by (format, path) ===" table
# exact rows should show 0%; direct_encode_sample rows show real extrapolation error
```

The `path` field is recorded in `bench/runner/modes/accuracy.py` and passed through to the per-iteration JSON. The `analyze_accuracy.py` script in `reports/full-bench/` groups rows by `(format, path)` and prints median/p95/max error for each bucket. The `path` field is `Optional[str]` in `schemas.py` — existing API clients that ignore unknown fields are unaffected.

## Common commands

```bash
# Build the canonical synthetic corpus
python -m bench.corpus build --manifest core

# Verify the on-disk corpus matches the manifest pixel-hashes
python -m bench.corpus verify --manifest core

# Pre-warm the fetcher cache (downloads all URLs, verifies hashes)
python -m bench.corpus fetch --manifest full

# Build the fetched corpus (uses cached files; fetches on cache miss)
python -m bench.corpus build --manifest full --seal   # first time: seal pixel hashes
python -m bench.corpus build --manifest full           # subsequent builds

# Verify the fetched corpus
python -m bench.corpus verify --manifest full

# Override fetcher cache location
python -m bench.corpus fetch --manifest full --cache /path/to/cache
python -m bench.corpus build --manifest full --cache /path/to/cache

# Quick smoke (1 iter, all formats, ~1 min)
python -m bench.run --mode quick

# Honest latency (5 iters, isolated, p50/p95/p99)
python -m bench.run --mode timing --manifest core --out reports/timing.json

# Peak RSS truth (1 iter, isolated)
python -m bench.run --mode memory --out reports/memory.json

# Diff two runs with Welch's t-test
python -m bench.compare reports/baseline.json reports/head.json --threshold-pct 10
```

## Local environment

### Enabling JXL locally

`settings.enable_jxl` defaults to `False` because the JXL optimizer requires libjxl binaries
(`cjxl`, `djxl`) and the `jxlpy` Python package, which are not present in every environment.
Without the flag, every JXL bench case fails with `UnsupportedFormatError: Format jxl is not enabled`.

If `cjxl`, `djxl`, and `jxlpy` are available in your venv, set `ENABLE_JXL=true` before running bench:

```bash
# Verify the toolchain is present
which cjxl djxl
.venv/bin/python -c "from jxlpy import JXLPyEncoder, JXLPyDecoder; print('ok')"

# One-off run
ENABLE_JXL=true python -m bench.run --mode quick --fmt jxl --manifest core --out /tmp/jxl.json

# Persistent (direnv) — copy .envrc.example to .envrc and run `direnv allow`
cp .envrc.example .envrc
direnv allow
```

`bench.run` emits a warning automatically when `enable_jxl=False` but both `cjxl` and `jxlpy`
are detected, so you won't silently get 21 failures without explanation.

**Deep-color JXL cases (10/12-bit)**: Even with `ENABLE_JXL=true`, the 6 deep-color JXL bench
cases (`deep_color_10bit_*` and `deep_color_12bit_*`) fail with
`NotImplementedError: bits_per_sample not equals 8`. This is a known limitation — the
`PillowReencodeOptimizer` base class opens images via `Image.open()` which decodes to 8-bit.
The 15 standard 8-bit JXL cases all succeed. Deep-color support is a separate deferred item.

## CI integration

The workflow `.github/workflows/bench-pr.yml` runs automatically on pull requests that touch `optimizers/`, `estimation/`, `utils/`, `bench/`, `schemas.py`, `requirements.txt`, `Dockerfile`, or the workflow file itself.

**Baseline location**: `reports/baseline.core.json` — checked into the repo (unignored via `.gitignore`).

**What CI does**:
1. Builds the Docker image from the PR's source tree (with Buildx GHA layer cache for speed).
2. Inside the container, runs `python -m bench.corpus build --manifest core` then `python -m bench.run --mode quick --manifest core --out reports/_head.json`.
3. Runs `python -m bench.compare reports/baseline.core.json reports/_head.json --threshold-pct 10 --format markdown`.
4. Posts (or updates) a PR comment with the diff table. A hidden HTML signature `<!-- pare-bench-comment -->` ensures repeat pushes update the same comment rather than creating new ones.
5. Fails the workflow if `bench.compare` exits non-zero (regression in at least one case exceeds ±10%).

**Threshold**: ±10% change in median `wall_ms`. Both Welch's t-test (α=0.05) and Cohen's d (≥0.5) must clear before a case is flagged — see `bench/runner/compare.py` for the full logic.

**No-baseline path**: If `reports/baseline.core.json` is missing, the workflow posts a "no baseline; this run is the candidate baseline" comment and does not fail.

**Auto-baseline-update on main merge**: `.github/workflows/bench-baseline-update.yml` runs `bench.run --mode quick --manifest core` after every merge to main. If the new run is statistically indistinguishable from the previous baseline (`bench.compare --threshold-pct 10` exits 0), the workflow commits the refreshed baseline as `chore(bench): refresh baseline.core.json [skip ci]`. If the comparison detects significant deltas, the workflow opens a `bench, drift` issue with the markdown diff for human triage rather than overwriting the baseline. This guards against gradual regressions normalizing into the baseline.

**How to refresh the baseline** (run locally, then commit the result):

```bash
python -m bench.corpus build --manifest core
python -m bench.run --mode quick --manifest core \
  --annotate "env=local-venv-bootstrap" \
  --out reports/baseline.core.json
git add reports/baseline.core.json
git commit -m "chore(bench): refresh baseline.core.json"
```

Refresh the baseline whenever you intentionally change optimizer behavior, add corpus entries, or when you want to adopt the Docker-built numbers as the new reference (run the CI workflow on a clean branch, pull `reports/_head.json` from the artifact, rename it, and commit).

## `bench compare` comparability guards

`bench.compare` refuses to diff runs that cannot produce meaningful results:

| check | behaviour | override |
|-------|-----------|----------|
| `metadata.mode` mismatch | **exits 2** with a clear error | `--allow-mismatched-mode` |
| `metadata.config.isolate` mismatch | **warns** to stderr | `--allow-mismatched-isolate` |
| `metadata.host.platform` mismatch | **warns** to stderr | `--allow-mismatched-platform` |

Motivation: a TIFF false-regression investigation traced to a baseline run in Linux Docker quick mode being diffed against a macOS isolated timing mode head — every TIFF case appeared broken. Mode mismatch is a hard error (exit 2) because `wall_ms` is not comparable across modes. Isolate and platform mismatches are warnings because they affect timings quantitatively but the direction of drift is predictable.

**Markdown output** always includes a `Compare conditions` header block showing mode/isolate/platform for both runs. **JSON output** includes `metadata.conditions.baseline` and `metadata.conditions.head` with the same fields.

```bash
# Normal compare — will exit 2 if modes differ
python -m bench.compare reports/baseline.json reports/head.json --threshold-pct 10

# Cross-mode rough trend (e.g. comparing quick vs timing as a sanity check only)
python -m bench.compare reports/baseline.json reports/head.json \
  --allow-mismatched-mode --allow-mismatched-platform

# Suppress all warnings for cross-environment CI comparisons
python -m bench.compare reports/baseline.json reports/head.json \
  --allow-mismatched-isolate --allow-mismatched-platform
```

## Dashboard

The bench dashboard at `https://amitray007.github.io/pare/` shows per-format timing/RSS trends
across the git history of `reports/baseline.core.json`. Regenerated on every main merge that
touches the baseline. Source: `bench/dashboard/build.py`.

The dashboard is purely informational — it doesn't gate any CI; for that, use `bench-pr.yml`.

**Memory measurements caveat:** `peak_rss_kb`, `parent_peak_rss_kb`, and `children_peak_rss_kb`
are only per-case isolated under `--mode memory`. Under `--mode quick` (`repeat=1`), these fields
reflect the process-wide RSS sampled at the end of each case's measurement window — there is no
process restart between cases, so all cases in a quick run share the same accumulating heap.
In practice this produces only 3–5 distinct RSS values across hundreds of cases (the RSS monotonically
grows through the run). Comparing per-case RSS values from a quick run is meaningless; the dashboard
annotates the "peak RSS" trend chart with a warning when any displayed run used quick mode.
Use `--mode memory` for per-case isolated RSS analysis.

```bash
# Build the dashboard locally
python -m bench.dashboard.build --out-dir /tmp/dash
# Inspect output
ls /tmp/dash                          # index.html, data/history.json
python -c "import json; d=json.loads(open('/tmp/dash/data/history.json').read()); print('runs:', len(d['runs']))"
```

## Code style

Same as repo root: Black 100 cols, Ruff E/F/W/I, Python 3.12, pytest-asyncio.

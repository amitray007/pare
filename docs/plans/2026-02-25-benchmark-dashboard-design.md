# Benchmark Health Dashboard — Design Doc

## Problem

The current benchmark system runs 44 photos x 3 sizes x ~7 formats x 3 presets = ~2,700+ cases, producing massive table dumps that make it hard to answer three simple questions:

1. Do presets differentiate correctly? (HIGH > MEDIUM > LOW reduction)
2. Is estimation accuracy within target? (<15% error)
3. Are there regressions vs. previous runs?

## Solution

A local web-based benchmark dashboard (`localhost:8081`) that:
- Picks 3 representative images per format (small/medium/large) from the corpus
- Runs optimization + estimation across all 3 presets
- Streams results live via SSE
- Renders a "mission control" style health dashboard
- Stores run history as flat JSON files for comparison

## Architecture

### Server

**File:** `benchmarks/server.py` — standalone FastAPI app on port 8081, separate from production Pare.

**Storage:** `.benchmark-data/runs/` at project root (gitignored). One JSON file per run: `run-<timestamp>.json`.

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET /` | Serve dashboard HTML |
| `GET /api/corpus` | List available formats + image counts |
| `POST /api/run` | Start a benchmark run, returns run ID |
| `GET /api/run/{id}/stream` | SSE stream of results |
| `GET /api/runs` | List past runs (history) |
| `GET /api/runs/{id}` | Full results of a past run |
| `DELETE /api/runs/{id}` | Delete a past run |

### Image Selection

For each format present in `tests/corpus/`:
- Pick 1 small, 1 medium, 1 large image (deterministic selection: first alphabetically per size tier)
- Configurable: 1, 3, or 5 images per format
- Formats auto-detected from corpus file extensions

### Benchmark Execution

Reuses `benchmarks/runner.run_single()` to run optimize + estimate per case.
Results streamed via SSE as each (format, preset) combination completes.
Semaphore-bounded parallelism (same as existing runner).

## UI Design

### Design Direction: "Mission Control"

Dark, industrial-utilitarian aesthetic. Information-dense but precise.

### Layout: Split panel

- **Left rail (~240px):** Run controls at top, history list below — always visible
- **Main area:** Dashboard results

### Typography

- Numbers: `JetBrains Mono` (technical precision)
- Labels: `DM Sans` (clean, doesn't compete with data)

### Color Palette

- Background: `#0a0e14`
- Surface: `#12171f`
- Borders: `#1e2530`
- Pass: `#34d399` (emerald)
- Warn: `#fbbf24` (amber)
- Fail: `#f87171` (coral)
- Accent: `#60a5fa` (sky blue)
- Text primary: `#e2e8f0`, secondary: `#64748b`

### Dashboard Sections (top to bottom in main area)

1. **Header bar:** "PARE BENCHMARK" title, [Configure v] drawer toggle, [Run All] green button
2. **Config drawer** (collapsed by default, slides down): Format chips (toggleable), preset pill buttons, images-per-format dropdown
3. **Health grid:** One card per format (120x80px) with format name, 3-bar micro sparkline (HIGH/MED/LOW), border-left color for pass/warn/fail. On hover: exact numbers.
4. **Preset comparison:** Grouped horizontal bar chart per format. 3 bars per group (HIGH/MED/LOW). Dashed overlay line shows estimation. Gap = estimation error.
5. **Estimation accuracy table + Worst outliers table:** Side by side at bottom.

### History Sidebar

- Each past run: date, row of colored dots (one per format health), duration
- Click to load into main area
- Delete with confirmation
- Compare mode: select two runs for delta view

### Streaming UX

- Format cards start as dark silhouettes
- Cards fill in with left-to-right wipe animation as SSE delivers results
- Progress indicator at top

## Pass/Fail Criteria

**Per-format health:**
- **Pass (green):** Presets differentiate (HIGH reduction > MEDIUM > LOW) AND avg estimation error < 10%
- **Warn (yellow):** Presets differentiate but estimation error 10-15%, OR presets don't differentiate but estimation is good
- **Fail (red):** Presets don't differentiate AND estimation error > 15%

## Storage Format

```json
{
  "id": "run-20260225-143022",
  "timestamp": "2026-02-25T14:30:22Z",
  "git_commit": "8030c1c",
  "config": {
    "formats": ["jpeg", "png", "webp", "gif", "bmp", "tiff"],
    "presets": ["HIGH", "MEDIUM", "LOW"],
    "images_per_format": 3
  },
  "duration_s": 45.2,
  "results": [ ... ],
  "health": {
    "jpeg": "pass",
    "png": "pass",
    "webp": "warn",
    ...
  }
}
```

## What This Does NOT Do

- Does not replace the existing full benchmark suite (`python -m benchmarks.run`)
- Does not touch the production Pare API
- Does not require any new Python dependencies (FastAPI + uvicorn already in the project)

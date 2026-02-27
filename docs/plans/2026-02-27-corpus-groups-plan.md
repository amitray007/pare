# Corpus Groups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the benchmark corpus into 4 technical groups (High-Res, Standard, Compact, Deep-Color) with a groups.json manifest, native format sourcing, and dashboard integration for group-level analysis.

**Architecture:** The `groups.json` file becomes the single source of truth for corpus organization, replacing filename-based tier classification. The server reads this manifest to select cases by group. The dashboard config dropdown gains group toggle buttons, and results carry group metadata for per-group breakdowns.

**Tech Stack:** Python 3.12, FastAPI, Pillow, Unsplash CDN (Imgix), external sample sites (heic.digital, libjxl, convertico)

---

### Task 1: Add `group` field to BenchmarkCase

**Files:**
- Modify: `benchmarks/cases.py:30-37`

**Step 1: Add the group field to BenchmarkCase**

In `benchmarks/cases.py`, add `group` field to the dataclass:

```python
@dataclass
class BenchmarkCase:
    name: str
    data: bytes
    fmt: str
    category: str  # size: small, medium, large, vector
    content: str  # content type: photo, screenshot, graphic, etc.
    quality: int = 0  # source quality for lossy formats
    group: str = ""  # corpus group: high_res, standard, compact, deep_color
```

**Step 2: Verify existing tests still pass**

Run: `pytest tests/ -x -q`
Expected: All tests pass (group defaults to empty string, no breaking change)

**Step 3: Commit**

```bash
git add benchmarks/cases.py
git commit -m "feat: add group field to BenchmarkCase dataclass"
```

---

### Task 2: Add CorpusGroup definitions to constants.py

**Files:**
- Modify: `benchmarks/constants.py`

**Step 1: Add CorpusGroup dataclass and CORPUS_GROUPS dict**

Append to `benchmarks/constants.py` after the `PRESETS_BY_NAME` line (line 73):

```python
# ---------------------------------------------------------------------------
# Corpus groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusGroup:
    key: str
    label: str
    badge: str  # Short label for UI badges
    description: str


CORPUS_GROUPS = {
    "high_res": CorpusGroup(
        key="high_res",
        label="High-Res",
        badge="Hi-Res",
        description="2000+ px, >500KB",
    ),
    "standard": CorpusGroup(
        key="standard",
        label="Standard",
        badge="Std",
        description="800-2000px, 100-500KB",
    ),
    "compact": CorpusGroup(
        key="compact",
        label="Compact",
        badge="Cmpct",
        description="<800px, <100KB",
    ),
    "deep_color": CorpusGroup(
        key="deep_color",
        label="Deep-Color",
        badge="Deep",
        description="10/12-bit native encoding",
    ),
}
```

**Step 2: Verify import works**

Run: `python -c "from benchmarks.constants import CORPUS_GROUPS; print(list(CORPUS_GROUPS.keys()))"`
Expected: `['high_res', 'standard', 'compact', 'deep_color']`

**Step 3: Commit**

```bash
git add benchmarks/constants.py
git commit -m "feat: add CorpusGroup definitions for 4 technical groups"
```

---

### Task 3: Create groups.json schema and loader

**Files:**
- Create: `benchmarks/corpus.py`

**Step 1: Write the corpus module**

Create `benchmarks/corpus.py` with functions to load and classify corpus images by group:

```python
"""Corpus management for benchmark groups.

Loads groups.json manifest and provides group-aware case selection.
Falls back to dimension-based classification when groups.json is absent.
"""

import json
from pathlib import Path

from benchmarks.cases import BenchmarkCase
from benchmarks.constants import CORPUS_GROUPS

# File extension to format mapping
_EXT_TO_FMT = {
    ".jpg": "jpeg", ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".gif": "gif",
    ".bmp": "bmp",
    ".tiff": "tiff", ".tif": "tiff",
    ".avif": "avif",
    ".heic": "heic", ".heif": "heic",
    ".jxl": "jxl",
    ".svg": "svg", ".svgz": "svgz",
}


def load_groups_manifest(corpus_dir: Path) -> dict | None:
    """Load groups.json from corpus directory. Returns None if not found."""
    manifest_path = corpus_dir / "groups.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _classify_group_by_dims(data: bytes, fmt: str, filepath: Path) -> str:
    """Classify an image into a group based on dimensions and path."""
    # Deep-color: files in specific native directories
    parts = filepath.parts
    if "deep_color" in parts or "avif_native" in parts:
        return "deep_color"

    if fmt in ("svg", "svgz"):
        return "standard"

    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        max_dim = max(img.size)
        file_size = len(data)

        if max_dim >= 2000 and file_size > 500_000:
            return "high_res"
        elif max_dim < 800:
            return "compact"
        else:
            return "standard"
    except Exception:
        # Fallback to file size
        size = len(data)
        if size > 500_000:
            return "high_res"
        if size < 100_000:
            return "compact"
        return "standard"


def load_corpus_cases(
    corpus_dir: Path,
    groups: list[str] | None = None,
    formats: list[str] | None = None,
) -> list[BenchmarkCase]:
    """Load corpus images as BenchmarkCase list, optionally filtered by group and format.

    If groups.json exists, uses it for group assignment.
    Otherwise falls back to dimension-based classification.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        return []

    manifest = load_groups_manifest(corpus_path)
    cases = []

    if manifest and "groups" in manifest:
        # Manifest-driven loading
        for group_key, group_data in manifest["groups"].items():
            if groups and group_key not in groups:
                continue
            for file_info in group_data.get("files", []):
                fmt = file_info.get("format", "")
                if formats and fmt not in formats:
                    continue
                filepath = corpus_path / file_info["path"]
                if not filepath.exists():
                    continue
                data = filepath.read_bytes()
                cases.append(BenchmarkCase(
                    name=f"{group_key}/{filepath.stem}",
                    data=data,
                    fmt=fmt,
                    category=file_info.get("category", "medium"),
                    content=filepath.parent.name,
                    group=group_key,
                ))
    else:
        # Fallback: scan directory and classify by dimensions
        for filepath in sorted(corpus_path.rglob("*")):
            if not filepath.is_file():
                continue
            ext = filepath.suffix.lower()
            fmt = _EXT_TO_FMT.get(ext)
            if fmt is None:
                continue
            if formats and fmt not in formats:
                continue

            data = filepath.read_bytes()
            group = _classify_group_by_dims(data, fmt, filepath)

            if groups and group not in groups:
                continue

            cases.append(BenchmarkCase(
                name=f"{filepath.parent.name}/{filepath.stem}",
                data=data,
                fmt=fmt,
                category=group,
                content=filepath.parent.name,
                group=group,
            ))

    return cases


def scan_corpus_by_group(
    corpus_dir: Path,
) -> dict[str, dict[str, list[Path]]]:
    """Scan corpus and group files by group and format.

    Returns: {group_key: {format: [paths]}}
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        return {}

    manifest = load_groups_manifest(corpus_path)
    result: dict[str, dict[str, list[Path]]] = {}

    if manifest and "groups" in manifest:
        for group_key, group_data in manifest["groups"].items():
            for file_info in group_data.get("files", []):
                fmt = file_info.get("format", "")
                filepath = corpus_path / file_info["path"]
                if filepath.exists():
                    result.setdefault(group_key, {}).setdefault(fmt, []).append(filepath)
    else:
        # Fallback scan
        for filepath in sorted(corpus_path.rglob("*")):
            if not filepath.is_file():
                continue
            ext = filepath.suffix.lower()
            fmt = _EXT_TO_FMT.get(ext)
            if fmt is None:
                continue
            data = filepath.read_bytes()
            group = _classify_group_by_dims(data, fmt, filepath)
            result.setdefault(group, {}).setdefault(fmt, []).append(filepath)

    return result
```

**Step 2: Verify module loads**

Run: `python -c "from benchmarks.corpus import load_corpus_cases, scan_corpus_by_group; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add benchmarks/corpus.py
git commit -m "feat: add corpus module with group-aware case loading"
```

---

### Task 4: Update server.py to use group-aware corpus

**Files:**
- Modify: `benchmarks/server.py:40-82` (imports, RunConfig)
- Modify: `benchmarks/server.py:131-224` (scan/select functions)
- Modify: `benchmarks/server.py:299-351` (endpoints)
- Modify: `benchmarks/server.py:384-406` (result_data in stream)

**Step 1: Update RunConfig to accept groups**

In `benchmarks/server.py`, change the `RunConfig` model (line 79-82):

```python
class RunConfig(BaseModel):
    formats: list[str] = []
    presets: list[str] = ["HIGH", "MEDIUM", "LOW"]
    images_per_format: int = 3
    groups: list[str] = []
```

**Step 2: Replace _scan_corpus and _select_cases with group-aware versions**

Replace `_scan_corpus()` (lines 134-168) and `_select_cases()` (lines 171-224) to use `benchmarks.corpus`:

```python
from benchmarks.corpus import load_corpus_cases as _load_corpus, scan_corpus_by_group

_corpus_cache: dict | None = None


def _scan_corpus() -> dict[str, dict[str, list[Path]]]:
    """Scan corpus by group. Returns: {group: {format: [paths]}}"""
    global _corpus_cache
    if _corpus_cache is not None:
        return _corpus_cache
    _corpus_cache = scan_corpus_by_group(CORPUS_DIR)
    return _corpus_cache


def _get_available_formats() -> dict[str, int]:
    """Get available formats with file counts."""
    corpus = _scan_corpus()
    fmt_counts: dict[str, int] = {}
    for group_data in corpus.values():
        for fmt, files in group_data.items():
            fmt_counts[fmt] = fmt_counts.get(fmt, 0) + len(files)
    return fmt_counts


def _select_cases(
    formats: list[str],
    groups: list[str],
    images_per_format: int,
) -> list[BenchmarkCase]:
    """Select cases from corpus filtered by groups and formats."""
    cases = _load_corpus(
        CORPUS_DIR,
        groups=groups if groups else None,
        formats=formats if formats else None,
    )
    # Limit per format
    by_fmt: dict[str, list[BenchmarkCase]] = {}
    for c in cases:
        by_fmt.setdefault(c.fmt, []).append(c)

    selected = []
    for fmt, fmt_cases in by_fmt.items():
        selected.extend(fmt_cases[:images_per_format])
    return selected
```

**Step 3: Update /api/corpus endpoint to return group info**

Replace the `/api/corpus` endpoint (lines 299-312):

```python
@app.get("/api/corpus")
async def get_corpus():
    """List available groups, formats, and counts."""
    corpus = _scan_corpus()
    groups_info = {}
    fmt_totals: dict[str, int] = {}

    for group_key, fmt_data in corpus.items():
        group_files = {}
        for fmt, files in fmt_data.items():
            group_files[fmt] = len(files)
            fmt_totals[fmt] = fmt_totals.get(fmt, 0) + len(files)
        groups_info[group_key] = {
            "total": sum(len(f) for f in fmt_data.values()),
            "formats": group_files,
        }

    # Add group labels from constants
    from benchmarks.constants import CORPUS_GROUPS
    for key, info in groups_info.items():
        if key in CORPUS_GROUPS:
            g = CORPUS_GROUPS[key]
            info["label"] = g.label
            info["badge"] = g.badge
            info["description"] = g.description

    return {
        "groups": groups_info,
        "formats": {fmt: {"total": count} for fmt, count in fmt_totals.items()},
        "corpus_dir": str(CORPUS_DIR),
    }
```

**Step 4: Update start_run to pass groups**

In the `start_run` endpoint (lines 316-351), update case selection:

```python
@app.post("/api/run")
async def start_run(config: RunConfig):
    """Start a benchmark run."""
    corpus = _scan_corpus()
    if not corpus:
        raise HTTPException(status_code=400, detail="No corpus found. Download it first.")

    fmt_counts = _get_available_formats()
    available_formats = list(fmt_counts.keys())
    formats = config.formats if config.formats else available_formats

    preset_names = [p.upper() for p in config.presets]
    for p in preset_names:
        if p not in PRESETS_BY_NAME:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {p}")

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    cases = _select_cases(formats, config.groups, config.images_per_format)
    if not cases:
        raise HTTPException(status_code=400, detail="No cases selected.")

    _active_runs[run_id] = {
        "config": config.model_dump(),
        "cases": cases,
        "preset_names": preset_names,
        "started": True,
    }

    return {
        "run_id": run_id,
        "cases_count": len(cases),
        "formats": formats,
        "presets": preset_names,
        "groups": config.groups or list(corpus.keys()),
        "total_tasks": len(cases) * len(preset_names),
    }
```

**Step 5: Add group to result_data in stream_run**

In the `stream_run` SSE handler, add `group` to result_data dict (around line 384-406):

Add this line after `"content": case.content,`:
```python
"group": case.group,
```

**Step 6: Verify server starts**

Run: `python -c "from benchmarks.server import app; print('Server imports OK')"`
Expected: `Server imports OK`

**Step 7: Commit**

```bash
git add benchmarks/server.py
git commit -m "feat: update server with group-aware corpus selection"
```

---

### Task 5: Update dashboard HTML — config dropdown

**Files:**
- Modify: `benchmarks/templates/dashboard.html:570-592` (config dropdown HTML)
- Modify: `benchmarks/templates/dashboard.html:628-641` (JS state + loadCorpus)
- Modify: `benchmarks/templates/dashboard.html:659-676` (JS functions)

**Step 1: Add group chips to config dropdown**

After the presets `cfg-group` div (line 581) and before the "Images per format" div (line 583), insert a new group:

```html
          <div class="cfg-group">
            <div class="cfg-label">Corpus Group</div>
            <div class="chip-grid" id="group-chips"></div>
          </div>
```

**Step 2: Update loadCorpus to store and render group info**

Update the `loadCorpus` function and add `renderGroupChips`:

```javascript
let corpusFormats = {}, corpusGroups = {}, currentResults = [], ...

async function loadCorpus() {
  try {
    const r = await fetch('/api/corpus');
    const d = await r.json();
    corpusFormats = d.formats;
    corpusGroups = d.groups || {};
    renderFormatChips();
    renderGroupChips();
  } catch(e) { console.error('Corpus load failed:', e); }
}

function renderGroupChips() {
  const c = document.getElementById('group-chips');
  if (!c) return;
  const allGroups = ['high_res', 'standard', 'compact', 'deep_color'];
  const labels = { high_res: 'High-Res', standard: 'Standard', compact: 'Compact', deep_color: 'Deep-Color' };
  c.innerHTML = allGroups.map(g => {
    const info = corpusGroups[g];
    const ok = info && info.total > 0;
    return `<button class="chip ${ok?'active':'unavailable'}" data-group="${g}" ${ok?'onclick="toggleChip(this)"':''} title="${ok?info.total+' files: '+info.description:'unavailable'}">${labels[g]}</button>`;
  }).join('');
}
```

**Step 3: Add getSelectedGroups and update runConfigured**

```javascript
function getSelectedGroups() { return [...document.querySelectorAll('#group-chips .chip.active')].map(c => c.dataset.group); }

async function runConfigured() {
  document.getElementById('config-drop').classList.remove('open');
  await startRun({
    formats: getSelectedFormats(),
    presets: getSelectedPresets(),
    images_per_format: parseInt(document.getElementById('images-per-format').value),
    groups: getSelectedGroups()
  });
}
```

**Step 4: Verify by starting server and loading dashboard**

Run: `python -m benchmarks.server`
Navigate to http://localhost:8081, open Config dropdown, verify group chips appear.

**Step 5: Commit**

```bash
git add benchmarks/templates/dashboard.html
git commit -m "feat: add corpus group selection to dashboard config"
```

---

### Task 6: Update dashboard — group badges and per-group breakdowns

**Files:**
- Modify: `benchmarks/templates/dashboard.html` — CSS (add badge styles) and JS (renderOverview, renderEstimation)

**Step 1: Add group badge CSS**

After the `.dpill` styles (around line 312), add:

```css
.g-badge {
  display: inline-flex; padding: 1px 6px; border-radius: 3px;
  font-size: 11px; font-weight: 600; font-family: var(--mono);
  background: var(--border-light); color: var(--secondary);
  margin-left: 4px;
}
.fc-groups {
  display: flex; gap: 8px; flex-wrap: wrap;
  margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-light);
  font-size: 12px; color: var(--muted);
}
.fc-groups span { font-family: var(--mono); font-weight: 600; color: var(--text); }
```

**Step 2: Update renderOverview to show per-group breakdown in format cards**

Inside the `renderCardGroup` function, after the fc-presets `</div>` and before the fc-footer div (around line 868), add per-group breakdown:

```javascript
// Per-group breakdown
const byGroup = {};
for (const r of c.results) {
  if (!r.opt_error && r.group) {
    byGroup[r.group] = byGroup[r.group] || [];
    byGroup[r.group].push(r.reduction_pct);
  }
}
const groupLabels = { high_res: 'Hi-Res', standard: 'Std', compact: 'Cmpct', deep_color: 'Deep' };
const groupEntries = Object.entries(byGroup);
if (groupEntries.length > 1) {
  out += '<div class="fc-groups">';
  for (const [g, vals] of groupEntries) {
    const avg = vals.reduce((a,b) => a+b, 0) / vals.length;
    out += `<span>${groupLabels[g] || g}: ${avg.toFixed(1)}%</span>`;
  }
  out += '</div>';
}
```

**Step 3: Update renderEstimation to show group badges**

In the estimation row rendering (around line 931), add group badge after the name:

Replace the e-name span:
```javascript
<span class="e-name">${esc(trunc(r.name, 28))}${r.group ? '<span class="g-badge">'+groupLabels[r.group]+'</span>' : ''}</span>
```

Move the `groupLabels` const to the top-level scope (outside any function) so both renderOverview and renderEstimation can use it.

**Step 4: Verify by running a benchmark in the dashboard**

Run server, execute a benchmark, check Overview cards show per-group breakdown and Estimation shows badges.

**Step 5: Commit**

```bash
git add benchmarks/templates/dashboard.html
git commit -m "feat: add group badges and per-group breakdowns to dashboard"
```

---

### Task 7: Rewrite download script for group-based corpus

**Files:**
- Create: `scripts/download_corpus.py` (new, replaces `download_unsplash_corpus.py`)

**Step 1: Write the new download script**

Create `scripts/download_corpus.py` that:
1. Downloads 12 Unsplash photos (3 per group A/B/C) in 4 CDN-native formats (JPEG, PNG, AVIF, WebP)
2. Downloads external native samples for Group D (HEIC from heic.digital, AVIF from link-u, JXL/GIF/BMP from convertico)
3. Generates `groups.json` manifest
4. Supports `--group`, `--force`, `--skip-external` flags

The script should:
- Use hardcoded Unsplash photo IDs (curated for each group) instead of search queries
- Download at group-appropriate sizes: high_res=2400px, standard=1200px, compact=400px
- Create the `tests/corpus/{group}/` directory structure
- Write `groups.json` with full metadata (path, format, source_type, width, height, size_bytes)

Key structure:

```python
# 3 curated photos per group (by Unsplash photo ID)
# These should be selected for diverse content within each group
GROUP_PHOTOS = {
    "high_res": [
        {"id": "<photo_id_1>", "name": "landscape_01", "width": 2400},
        {"id": "<photo_id_2>", "name": "texture_02", "width": 2400},
        {"id": "<photo_id_3>", "name": "architecture_03", "width": 2400},
    ],
    "standard": [
        {"id": "<photo_id_4>", "name": "portrait_01", "width": 1200},
        {"id": "<photo_id_5>", "name": "food_02", "width": 1200},
        {"id": "<photo_id_6>", "name": "macro_03", "width": 1200},
    ],
    "compact": [
        {"id": "<photo_id_7>", "name": "abstract_01", "width": 400},
        {"id": "<photo_id_8>", "name": "monochrome_02", "width": 400},
        {"id": "<photo_id_9>", "name": "colorful_03", "width": 400},
    ],
}

# CDN formats to download per photo
CDN_FORMATS = [("jpg", "jpg", "q=90"), ("png", "png", ""), ("avif", "avif", "q=80"), ("webp", "webp", "q=90")]

# External native samples for deep_color group
EXTERNAL_SAMPLES = {
    "avif_native": [...],  # link-u samples (already defined)
    "heic_native": [...],  # heic.digital URLs
    "jxl_native": [...],   # libjxl testdata or convertico
    "gif_native": [...],   # convertico animated GIFs
    "bmp_native": [...],   # convertico BMP samples
}
```

Note: The exact Unsplash photo IDs should be selected by running `--dry-run` against the existing corpus manifest to find good candidates, OR by selecting from the existing downloaded photos.

**Step 2: Test with --dry-run**

Run: `python scripts/download_corpus.py --dry-run`
Expected: Lists all files that would be downloaded without downloading

**Step 3: Commit**

```bash
git add scripts/download_corpus.py
git commit -m "feat: add group-based corpus download script"
```

---

### Task 8: Update convert script for group-based corpus

**Files:**
- Modify: `scripts/convert_corpus_formats.py`

**Step 1: Update convert script to work with group directories**

Update the script to:
- Scan `tests/corpus/{high_res,standard,compact}/` for PNG files
- Convert PNG -> BMP, TIFF, GIF, HEIC, JXL (only for files that don't exist)
- Update `groups.json` with converted files (source_type: "lossless")
- Skip `deep_color/` directory (those are all native samples)

**Step 2: Test conversion**

Run: `python scripts/convert_corpus_formats.py --dry-run`
Expected: Shows which conversions would be performed

**Step 3: Commit**

```bash
git add scripts/convert_corpus_formats.py
git commit -m "feat: update convert script for group-based corpus structure"
```

---

### Task 9: Update CLI benchmark (run.py) with --group flag

**Files:**
- Modify: `benchmarks/run.py:131-155` (argparse)
- Modify: `benchmarks/run.py:179-195` (corpus loading)

**Step 1: Add --group argument**

In `benchmarks/run.py`, add after the `--corpus` argument:

```python
parser.add_argument(
    "--group",
    nargs="+",
    choices=["high_res", "standard", "compact", "deep_color"],
    help="Filter corpus by group (only with --corpus)",
)
```

**Step 2: Update corpus loading to use groups**

When loading corpus cases, pass the group filter:

```python
if args.corpus:
    from benchmarks.corpus import load_corpus_cases as load_grouped_corpus
    corpus_cases = load_grouped_corpus(
        Path(args.corpus),
        groups=args.group,
    )
    if not corpus_cases:
        parser.error(f"No image files found in corpus directory: {args.corpus}")
    print(f"  Loaded {len(corpus_cases)} images from corpus: {args.corpus}", file=sys.stderr)
    if args.group:
        print(f"  Groups: {', '.join(args.group)}", file=sys.stderr)
```

**Step 3: Verify**

Run: `python -m benchmarks.run --corpus tests/corpus --group high_res --fmt jpeg --no-save`
Expected: Only runs high_res JPEG cases

**Step 4: Commit**

```bash
git add benchmarks/run.py
git commit -m "feat: add --group flag to CLI benchmark"
```

---

### Task 10: Download corpus and generate groups.json

**Step 1: Run the download script**

First, source the Unsplash API key:
```bash
source .env
python scripts/download_corpus.py
```

**Step 2: Run the convert script to fill format gaps**

```bash
python scripts/convert_corpus_formats.py
```

**Step 3: Verify groups.json was generated**

```bash
python -c "import json; d = json.load(open('tests/corpus/groups.json')); print({k: len(v['files']) for k, v in d['groups'].items()})"
```

Expected: Shows file counts per group

**Step 4: Commit groups.json (NOT the image files — they're gitignored)**

```bash
git add tests/corpus/groups.json
git commit -m "feat: add groups.json manifest for corpus organization"
```

---

### Task 11: Clean up old corpus files

**Step 1: Remove old category-based directories**

After verifying the new group-based corpus works, remove the old 13 category directories:

```bash
# Verify new corpus works first
python -m benchmarks.run --corpus tests/corpus --no-save --fmt jpeg

# Then remove old directories (backup first)
# abstract, aerial, architecture, colorful, food, highcontrast, landscape,
# lowlight, macro, monochrome, portrait, text_heavy, texture
```

**Step 2: Remove old manifest.json**

```bash
rm tests/corpus/manifest.json
```

**Step 3: Delete old download script**

```bash
git rm scripts/download_unsplash_corpus.py
```

**Step 4: Update run_corpus_benchmark.py to not need --corpus flag**

Update `scripts/run_corpus_benchmark.py` to use the new corpus module directly.

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: clean up old corpus structure, remove category directories"
```

---

### Task 12: Full integration test

**Step 1: Run full CLI benchmark with corpus**

```bash
python -m benchmarks.run --corpus tests/corpus
```

Expected: All groups tested, results show group metadata

**Step 2: Run benchmark server test**

```bash
python -m benchmarks.server
# In browser: http://localhost:8081
# Click Config -> verify group chips
# Run a benchmark -> verify group badges and per-group breakdowns
```

**Step 3: Run all tests**

```bash
pytest tests/ -x -q
```

Expected: All tests pass

**Step 4: Run linting**

```bash
python -m ruff check . && python -m black --check .
```

**Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: integration test fixes for corpus groups"
```

---

### Task 13: Update CLAUDE.md and benchmarks/CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `benchmarks/CLAUDE.md`

**Step 1: Update root CLAUDE.md**

Add corpus group commands to the Common Commands section and update architecture docs.

**Step 2: Update benchmarks/CLAUDE.md**

Add group info to Running Benchmarks, Architecture, and Key Metrics sections.

**Step 3: Commit**

```bash
git add CLAUDE.md benchmarks/CLAUDE.md
git commit -m "docs: update CLAUDE.md files with corpus group documentation"
```

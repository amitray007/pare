# Corpus Groups Design

## Problem

The benchmark corpus has 1,281 files (2 GB) but benchmarks use at most 45 files per run. Images aren't organized by technical profile, making it hard to answer "does optimization work for large images?" vs "does estimation work for thumbnails?" The corpus also relies heavily on format conversion from JPEG, which introduces recompression artifacts for lossy formats.

## Design

### 4 Technical Groups

Each group has a distinct identity that tests a different aspect of the optimization pipeline.

| Group | Key | Label | Criteria | What It Tests |
|-------|-----|-------|----------|---------------|
| A | `high_res` | High-Res | Longest dim >= 2000px, file > 500KB | Sample-based estimation accuracy; large-file optimization savings |
| B | `standard` | Standard | Longest dim 800-2000px, file 100-500KB | Production sweet spot; real-world web optimization |
| C | `compact` | Compact | Longest dim < 800px, file < 100KB | Exact-mode estimation; optimization overhead vs benefit |
| D | `deep_color` | Deep-Color | 10/12-bit native encoding profiles | Format edge cases; native modern format handling |

### Corpus Structure

```
tests/corpus/
  groups.json              # Group definitions + file inventory
  high_res/                # Group A: 3 photos x 9 formats = 27 files
  standard/                # Group B: 3 photos x 9 formats = 27 files
  compact/                 # Group C: 3 photos x 9 formats = 27 files
  deep_color/              # Group D: Native samples only
    avif_native/           # link-u 10/12-bit AVIF (8 files)
    heic_native/           # heic.digital iPhone/iPad samples
    jxl_native/            # libjxl testdata samples
    gif_native/            # convertico animated GIFs
    bmp_native/            # convertico BMP samples (various bit depths)
```

Total: ~101 files, ~145 MB (down from 1,281 files / 2 GB).

### Format Sources

| Format | Source | Source Type |
|--------|--------|------------|
| JPEG | Unsplash CDN `fm=jpg` | `cdn` |
| PNG | Unsplash CDN `fm=png` | `cdn` |
| AVIF | Unsplash CDN `fm=avif` | `cdn` |
| WebP | Unsplash CDN `fm=webp` | `cdn` |
| HEIC | heic.digital (native iPhone/iPad/Samsung) + PNG-source | `native` / `lossless` |
| JXL | libjxl/testdata (CC-BY-4.0) + PNG-source | `native` / `lossless` |
| GIF | convertico (animated) + PNG-source | `native` / `lossless` |
| BMP | convertico (various bit depths) + PNG-source | `native` / `lossless` |
| TIFF | PNG-source (lossless-to-lossless) | `lossless` |

Each file is tagged with `source_type` in `groups.json`:
- `cdn` = Unsplash CDN native encoding from original
- `native` = External native sample (real device/encoder output)
- `lossless` = Converted from PNG (lossless, bit-perfect for lossless formats)

### groups.json Schema

```json
{
  "version": 1,
  "groups": {
    "high_res": {
      "label": "High-Res",
      "badge": "Hi-Res",
      "description": "2000+ px, >500KB — tests sample-based estimation and large-file optimization",
      "files": [
        {
          "path": "high_res/landscape_01.jpg",
          "format": "jpeg",
          "source_type": "cdn",
          "width": 2400,
          "height": 1800,
          "size_bytes": 1752000
        }
      ]
    }
  }
}
```

### Download Script (scripts/download_corpus.py)

Renamed from `download_unsplash_corpus.py`. Changes:
- Downloads 12 specific Unsplash photos (3 per group A/B/C, by photo ID) instead of searching 13 categories
- Downloads in 4 CDN-native formats per photo: JPEG, PNG, AVIF, WebP
- Downloads external native samples for Group D (HEIC, JXL, GIF, BMP)
- Generates `groups.json` manifest with full metadata
- Flags: `--group`, `--force`, `--skip-external`

### Convert Script (scripts/convert_corpus_formats.py)

Updated:
- Only converts PNG -> BMP, TIFF, GIF, HEIC, JXL for groups A/B/C (fill gaps)
- Tags converted files with `source_type: "lossless"` in groups.json
- Skips formats that already have native files

### Server Changes (benchmarks/server.py)

- `RunConfig` gains `groups: list[str] = []` (empty = all groups)
- `_scan_corpus()` reads `groups.json` instead of filename-based tier classification
- `_select_cases()` filters by selected groups
- Each result carries `group` field (high_res/standard/compact/deep_color)
- `/api/corpus` response includes group info with file counts

### Dashboard Changes (benchmarks/templates/dashboard.html)

**Config dropdown** — new "Corpus Group" section with toggle buttons:
```
[ High-Res ] [ Standard ] [ Compact ] [ Deep-Color ]
```
All active by default. Sent as `groups` array in RunConfig.

**Overview tab** — format cards gain per-group breakdown:
```
By group: Hi-Res: 45.2%  Std: 38.1%  Cmpct: 12.7%
```

**Estimation tab** — results organized by group with group badges next to each result name.

**Summary strip** — "Groups tested" card showing which groups were included.

### CLI Changes (benchmarks/run.py)

- `--group` flag to filter by group when using `--corpus`
- Auto-detects `groups.json` in corpus directory

### Key Design Decisions

1. **12 unique photos** (no reuse across groups) ensures each group tests genuinely different images
2. **Hybrid sourcing** — CDN-native for 4 formats, external native for HEIC/JXL/GIF/BMP, PNG-source conversion for gaps
3. **source_type tagging** — full transparency about how each file was obtained
4. **Groups in config dropdown** — integrates with existing UI pattern, no new tabs needed
5. **groups.json as single source of truth** — replaces filename-based classification

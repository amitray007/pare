# Pare: Image Compression Service

A serverless image compression and optimization service built on Google Cloud Platform. Supports 12 image formats with format-specific optimization pipelines and a fast estimation engine.

## Supported Formats

| Format | Tool/Library | Strategy |
|--------|-------------|----------|
| PNG/APNG | pngquant + oxipng | Lossy quantization + lossless recompression |
| JPEG | Pillow/jpegli + jpegtran | Lossy re-encode + lossless Huffman optimization |
| WebP | cwebp + Pillow | Quality-based lossy/lossless encoding |
| GIF | gifsicle | Lossy/lossless with color reduction |
| SVG/SVGZ | Scour | Attribute/element cleanup, precision reduction |
| AVIF | pillow-avif-plugin | Quality-based AV1 encoding |
| HEIC | pillow-heif | Quality-based HEVC encoding |
| TIFF | Pillow | Adobe Deflate, LZW, or JPEG-in-TIFF (parallel) |
| BMP | Pillow + custom RLE8 | Palette quantization + content-aware RLE8 compression |
| JXL | jxlpy (pillow-jxl-plugin) | Quality-based JPEG XL re-encoding |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8080

# Or with Docker (includes all CLI tools + Redis)
docker-compose up
```

## API

### POST /optimize

Compress an image. Accepts multipart upload or JSON with URL.

```bash
# Multipart file upload
curl -X POST http://localhost:8080/optimize \
  -F "file=@photo.png" \
  -F 'options={"quality": 60, "strip_metadata": true}' \
  --output optimized.png

# URL-based with storage
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/photo.png", "optimization": {"quality": 60}}'
```

### POST /estimate

Predict compression savings without running the full optimizer (~20-50ms).

```bash
curl -X POST http://localhost:8080/estimate -F "file=@photo.png"
```

### GET /health

Check service status and tool availability.

## Configuration

All settings via environment variables (see `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `WORKERS` | 4 | Uvicorn workers |
| `MAX_FILE_SIZE_MB` | 32 | Upload size limit |
| `DEFAULT_QUALITY` | 80 | Default quality (1-100) |
| `API_KEY` | "" | Bearer token (empty = no auth) |
| `REDIS_URL` | "" | Redis for rate limiting |
| `ALLOWED_ORIGINS` | "*" | CORS origins |

## Quality Presets

Lower quality = more aggressive compression, smaller files.

| Preset | Quality | Behavior |
|--------|---------|----------|
| HIGH | ~40 | Lossy methods, aggressive quantization |
| MEDIUM | ~60 | Lossy methods, moderate settings |
| LOW | ~80 | Lossless-only where possible |

## Benchmarks

```bash
python -m benchmarks.run                          # All formats, all presets
python -m benchmarks.run --fmt png --preset high  # Filter
python -m benchmarks.run --compare                # Delta vs previous run
python -m benchmarks.run --json                   # JSON to stdout
```

Reports are saved to `reports/` as timestamped HTML + JSON files.

### Benchmark Results by Format

Results from 369 test cases across all formats and presets. Quality presets: **HIGH** (q=40, aggressive lossy), **MEDIUM** (q=60, moderate lossy), **LOW** (q=80, lossless-preferred).

#### PNG

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 60.6% | 14.6% - 95.0% | 0.3% | pngquant + oxipng |
| MEDIUM | 58.7% | 14.6% - 95.0% | 0.2% | pngquant + oxipng |
| LOW | 42.1% | 0.0% - 95.0% | 0.1% | oxipng (lossless only) |

**Content breakdown (HIGH):** photo 77.3%, screenshot 65.0%, graphic 59.6%, transparent 58.0%, solid 84.7%, gradient 19.1%, palette 40.0%

**Special conditions:**
- HIGH/MEDIUM use lossy pngquant quantization followed by lossless oxipng recompression
- LOW uses oxipng lossless-only (no quantization); returns original if no savings
- Solid-color images compress extremely well (~85%); gradients compress poorly (~19%)
- Small files (<12KB) trigger exact probes for better estimation accuracy

#### JPEG

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 53.5% | 18.2% - 70.0% | 8.6% | jpegli, jpegtran |
| MEDIUM | 34.3% | 15.2% - 58.3% | 7.5% | jpegli, jpegtran |
| LOW | 20.7% | 11.5% - 58.3% | 1.7% | jpegli, jpegtran |

**Content breakdown (HIGH):** photo 52.9%, screenshot 65.5%

**Special conditions:**
- Jpegli (lossy re-encode) and jpegtran (lossless Huffman) run concurrently; smallest wins
- LOW preset still achieves ~20% via lossless Huffman optimization (jpegtran)
- Screenshots compress better than photos due to larger flat regions
- Source JPEG quality affects savings — high-quality sources (q=95) yield more reduction
- In Docker, uses jpegli (libjpeg from libjxl); locally falls back to libjpeg-turbo

#### WebP

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 48.5% | 21.0% - 69.7% | 2.5% | pillow (lossy) |
| MEDIUM | 28.2% | 6.0% - 48.9% | 2.4% | pillow (lossy) |
| LOW | 8.9% | 0.0% - 24.6% | 0.8% | pillow (lossless) / none |

**Special conditions:**
- Uses Pillow's WebP encoder with quality-mapped settings
- LOW preset returns original if lossless re-encode doesn't shrink the file
- Already-optimized WebP files may see 0% reduction at LOW preset

#### GIF

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 27.8% | 7.1% - 49.1% | 4.1% | gifsicle --lossy=80 --colors=128 |
| MEDIUM | 14.9% | 7.1% - 19.1% | 3.2% | gifsicle --lossy=30 --colors=192 |
| LOW | 7.7% | 0.0% - 18.8% | 2.9% | gifsicle (lossless) / none |

**Content breakdown (HIGH):** gradient 41.2%, graphic 14.3%

**Special conditions:**
- Gradient GIFs compress better (~41%) due to smooth color transitions amenable to color reduction
- Graphic GIFs with many distinct colors see less savings (~14%)
- LOW uses lossless gifsicle optimization; returns original if no savings
- Animated GIFs are supported (gifsicle preserves animation frames)

#### SVG

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 24.5% | 4.1% - 55.1% | 2.7% | scour |
| MEDIUM | 24.5% | 4.1% - 55.1% | 2.2% | scour |
| LOW | 12.1% | 0.0% - 33.6% | 2.1% | scour / none |

**Content breakdown (HIGH):** bloated 55.1%, simple 14.2%, complex 4.1%

**Special conditions:**
- HIGH and MEDIUM produce identical results (scour applies same XML cleanup)
- Bloated SVGs with redundant attributes, comments, or metadata see high reduction (~55%)
- Complex SVGs with minimal redundancy see very low reduction (~4%)
- LOW uses conservative scour settings; returns original for already-clean SVGs

#### SVGZ (Compressed SVG)

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 9.0% | 0.7% - 23.1% | 1.9% | scour |
| MEDIUM | 9.0% | 0.7% - 23.1% | 1.7% | scour |
| LOW | 6.2% | 0.0% - 18.6% | 3.9% | scour / none |

**Special conditions:**
- SVGZ is already gzip-compressed, so reductions are much smaller than SVG
- Scour cleans the XML, then re-compresses; gains come from smaller post-cleanup gzip
- Complex SVGZ files see minimal improvement (~0.7%)

#### AVIF

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 47.4% | 8.6% - 75.1% | 1.5% | avif-reencode |
| MEDIUM | 25.2% | 0.0% - 54.6% | 0.1% | avif-reencode / none |
| LOW | 7.7% | 0.0% - 23.2% | 0.1% | avif-reencode / none |

**Special conditions:**
- Re-encodes AVIF at target quality using pillow-avif-plugin
- Uses bpp-based estimation model calibrated to AV1 encoder curves
- At MEDIUM/LOW, already-efficient AVIF files may see 0% reduction (returned as-is)
- Estimation is highly accurate (<1.5% error) due to bpp-based prediction

#### HEIC

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 37.8% | 0.1% - 63.4% | 0.2% | heic-reencode |
| MEDIUM | 17.1% | 0.0% - 37.1% | 0.1% | heic-reencode / none |
| LOW | 1.8% | 0.0% - 5.4% | 0.1% | heic-reencode / none |

**Special conditions:**
- Re-encodes HEIC at target quality using pillow-heif
- HEVC encoding is generally less compressible than AV1 (AVIF), hence lower reductions
- LOW preset sees very minimal savings (~1.8%) since HEIC is already efficient
- Estimation uses bpp-based model similar to AVIF

#### TIFF

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 80.9% | 55.7% - 99.3% | 3.4% | tiff_jpeg, tiff_adobe_deflate |
| MEDIUM | 78.5% | 55.7% - 99.3% | 3.4% | tiff_jpeg, tiff_adobe_deflate |
| LOW | 42.2% | 0.0% - 98.8% | 20.4% | tiff_adobe_deflate / none |

**Content breakdown (HIGH):** photo 83.5%, screenshot 82.0%, graphic 77.2%

**Special conditions:**
- Compression methods (deflate, LZW, JPEG-in-TIFF) run in parallel via `asyncio.gather`
- HIGH/MEDIUM enable JPEG-in-TIFF for RGB/grayscale images (lossy, high reduction)
- LOW uses lossless-only (deflate/LZW); uncompressed TIFFs see very high savings (~98%)
- LOW estimation error is higher (20.4%) due to difficulty predicting lossless compression ratios across varied content
- Screenshots with large flat regions compress extremely well in all presets

#### BMP

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 87.8% | 66.1% - 98.6% | 9.5% | bmp-rle8, pillow-bmp-palette |
| MEDIUM | 69.2% | 66.1% - 75.0% | 0.3% | pillow-bmp-palette |
| LOW | 8.3% | 0.0% - 25.0% | 0.0% | pillow-bmp / none |

**Content breakdown (HIGH):** graphic 98.6%, screenshot 98.6%, photo 66.1%

**Special conditions:**
- HIGH uses content-aware RLE8 compression + palette quantization (best picked)
- RLE8 bonus is scaled by `flat_pixel_ratio`: screenshots/graphics (~98.6%) vs photos (~66.1%)
- MEDIUM uses palette quantization only (no RLE8), achieving consistent ~69% reduction
- LOW returns original for 24-bit BMPs unless palette reduction helps; often 0% savings
- BMP is an uncompressed format, so even moderate optimization yields high reductions

#### JXL (JPEG XL)

*Requires jxlpy or pillow-jxl-plugin. Tests and benchmarks skip when neither is installed; fully functional in Docker.*

| Preset | Avg Reduction | Range | Avg Est. Error | Methods |
|--------|--------------|-------|----------------|---------|
| HIGH | 55.0% | 2.2% - 92.2% | 0.9% | jxl-reencode |
| MEDIUM | 35.8% | 0.0% - 81.1% | 0.9% | jxl-reencode / none |
| LOW | 8.8% | 0.0% - 26.4% | 0.9% | jxl-reencode / none |

**Special conditions:**
- Re-encodes JXL at target quality using pillow-jxl-plugin or jxlpy
- Quality mapping: `jxl_quality = max(30, min(95, quality + 10))`
- Estimation uses bpp-based model calibrated to encoder output (0.9% avg error)
- High-quality sources (q=95) see massive reduction (~92% at HIGH) since JXL is very efficient
- Already-compressed JXL files (q=50) see minimal savings at any preset
- Format detection supports both bare codestream (`\xFF\x0A`) and ISOBMFF container
- In Docker, cjxl/djxl CLI tools are also available

### Estimation Accuracy Summary

| Format | HIGH Error | MEDIUM Error | LOW Error |
|--------|-----------|-------------|----------|
| PNG | 0.3% | 0.2% | 0.1% |
| JPEG | 8.6% | 7.5% | 1.7% |
| WebP | 2.5% | 2.4% | 0.8% |
| GIF | 4.1% | 3.2% | 2.9% |
| SVG | 2.7% | 2.2% | 2.1% |
| SVGZ | 1.9% | 1.7% | 3.9% |
| AVIF | 1.5% | 0.1% | 0.1% |
| HEIC | 0.2% | 0.1% | 0.1% |
| TIFF | 3.4% | 3.4% | 20.4% |
| BMP | 9.5% | 0.3% | 0.0% |
| JXL | 0.9% | 0.9% | 0.9% |

Target: <15% average error across all formats and presets. Most formats are well under target. TIFF LOW (20.4%) is an outlier due to difficulty predicting lossless compression ratios for varied content types.

## Deployment

Deploys to Cloud Run via Cloud Build (`cloudbuild.yaml`). The Dockerfile builds libjxl from source (providing jpegli for JPEG encoding and cjxl/djxl for JXL support) and installs all CLI tools.

```bash
docker build -t pare .
```

## License

[MIT](LICENSE)

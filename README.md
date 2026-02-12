# Pare: Image Compression Service

A serverless image compression and optimization service built on Google Cloud Platform. Supports 11 image formats with format-specific optimization pipelines and a fast estimation engine.

## Supported Formats

| Format | Tool/Library | Strategy |
|--------|-------------|----------|
| PNG/APNG | pngquant + oxipng | Lossy quantization + lossless recompression |
| JPEG | MozJPEG (cjpeg + jpegtran) | Lossy re-encode + lossless Huffman optimization |
| WebP | cwebp | Quality-based lossy/lossless encoding |
| GIF | gifsicle | Lossy/lossless with color reduction |
| SVG/SVGZ | Scour | Attribute/element cleanup, precision reduction |
| AVIF | pillow-heif | Quality-based AV1 encoding |
| HEIC | pillow-heif | Quality-based HEVC encoding |
| TIFF | Pillow | Adobe Deflate, LZW, or JPEG-in-TIFF |
| BMP | Pillow + custom RLE8 | Palette quantization + RLE8 compression |

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
```

Reports are saved to `reports/` as HTML and JSON.

## Deployment

Deploys to Cloud Run via Cloud Build (`cloudbuild.yaml`). The Dockerfile builds MozJPEG from source and installs all CLI tools.

```bash
docker build -t pare .
```

## License

[MIT](LICENSE)

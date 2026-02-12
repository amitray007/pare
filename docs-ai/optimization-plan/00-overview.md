# Pare Optimization Plan: Overview

## Context

Benchmarking (351 cases, 0 failures) revealed critical performance issues:

| Format | Problem | Worst Case Latency |
|--------|---------|-------------------|
| JPEG | MozJPEG + BMP decode + binary search | **40+ seconds** on 1920x1080 |
| PNG | oxipng level=6 + pngquant speed=1 | **20+ seconds** on 1920x1080 |
| WebP | Pillow method=6, synchronous, binary search | **35-105 seconds** worst case |
| AVIF | Metadata stripping only (no-op) | N/A — no real optimization |
| HEIC | Metadata stripping only (no-op) | N/A — no real optimization |
| GIF | Missing `--colors` flag | Only ~10% avg reduction |

**Root cause**: Python is NOT the bottleneck (1-2% of total time). The issues are tool selection, configuration, and architectural decisions around how tools are invoked.

## Phases

| Phase | File | Effort | Expected Impact |
|-------|------|--------|-----------------|
| 1 | [01-quick-wins.md](./01-quick-wins.md) | 1-2 days | PNG 20s→2-5s, GIF 10%→30-50%, WebP 2-3x faster |
| 2 | [02-event-loop-fixes.md](./02-event-loop-fixes.md) | 1 day | Concurrent request throughput 2-4x better |
| 3 | [03-avif-heic-encoding.md](./03-avif-heic-encoding.md) | 2-3 days | AVIF/HEIC from ~5% to 30-60% reduction |
| 4 | [04-jpegli-migration.md](./04-jpegli-migration.md) | 2-3 days | JPEG 40s→2-8s AND better compression |
| 5 | [05-future-enhancements.md](./05-future-enhancements.md) | Ongoing | JPEG XL, libvips migration, content-aware |

## Expected Scorecard After All Phases

| Area | Current | After Phase 1-2 | After Phase 3-4 | After Phase 5 |
|------|---------|-----------------|-----------------|---------------|
| JPEG latency (1080p) | 40s+ | 40s+ | **2-8s** | <2s |
| PNG latency (1080p) | 20s+ | **2-5s** | 2-5s | <2s |
| WebP latency (1080p) | 5-15s | **2-5s** | 2-5s | <1s |
| AVIF reduction | ~5% | ~5% | **30-60%** | 30-60% |
| HEIC reduction | ~5% | ~5% | **25-50%** | 25-50% |
| GIF reduction | ~10% | **30-50%** | 30-50% | 30-50% |
| Concurrent throughput | Blocked by sync ops | **2-4x better** | 2-4x better | 5-10x better |

## Guiding Principles

1. **Measure before and after** — Run `python -m benchmarks.run --fmt <format>` after each change
2. **Estimation must match optimizer** — Update `heuristics.py` whenever optimizer behavior changes
3. **Output guarantee preserved** — `_build_result()` in `base.py` always enforces output <= input
4. **No language rewrite needed** — Python overhead is 1-2%; tool/library choices are what matter

"""Benchmark run modes.

quick    — 1 iteration per case, all formats, ~1 minute. PR sanity check.
timing   — 5 iterations + 1 warmup, optional `--isolate` (per-case fresh
           subprocess for clean per-case parent RSS), p50/p95/p99 + MAD.
memory   — 1 iteration with tracemalloc + RSS curve sampling;
           max(parent, children) peak RSS as the headline number.
accuracy — runs `/estimate` and `/optimize` on each case; records
           predicted-vs-actual reduction error per case.
quality  — pure-numpy SSIM/PSNR + ssimulacra2/butteraugli on lossy
           outputs; `--quality-fast` skips the subprocess metrics.
load     — N concurrent requests per case through a fresh CompressionGate;
           measures throughput, 503 rate, latency tail under contention.
pr       — combined timing + accuracy + quality in a single pass per case.
           Default mode for the manual bench.yml workflow. Produces one
           JSON with all three signal categories per case.
"""

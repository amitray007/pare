"""Benchmark run modes.

quick   — 1 iteration per case, all formats, ~1 minute. PR sanity check.
timing  — 5 iterations + 1 warmup, --isolate, p50/p95/p99 + MAD.
memory  — 1 iteration + --isolate, max(parent, children) peak RSS as headline.
load    — deferred to v1 (different output shape: timeline, not per-case).
"""

"""Deterministic corpus builder for image compression benchmarks.

Synthesizers produce identical bytes from a (kind, seed, dimensions) triple.
Manifests pin pixel-level SHA-256 (raw RGB bytes, not encoded), since
encoded byte output drifts across libjpeg-turbo SIMD paths and platforms.
"""

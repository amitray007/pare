"""Optional real-content fetchers (v1).

Real fetchers (Kodak Lossless Suite, link-u/avif-sample-images for deep
color edges, Unsplash for ecological validity) are deferred to v1. The
synthetic suite alone covers every format x bucket cell, including the
8 pathological cases — enough to ship an honest benchmark today.

When implemented, fetchers will:

- Use hash-pinned URLs only (no random queries — pins reproducibility).
- Warn-and-skip if an API key is missing rather than fail the build.
- Cache to `bench/corpus/cache/` with atomic writes.
"""

"""Seeded synthetic image generators.

Every generator is a pure function of (seed, dimensions, params): re-running
with the same inputs produces byte-identical raw pixel data. The pixel
SHA-256 is the determinism contract — encoded outputs may vary across
libjpeg-turbo / libpng builds, so the manifest pins decoded pixel bytes.

Generators always take an explicit `random.Random(seed)` instance — never
mutate the global PRNG, since other libraries (e.g. PIL) also use it.

Synthesizers register themselves with `register_kind()` at import time;
`synthesize()` dispatches a `ManifestEntry` to the correct generator.
"""

from __future__ import annotations

from bench.corpus.manifest import ManifestEntry, Synthesized

# Importing for side effects (registration). Order doesn't matter.
from bench.corpus.synthesis import (  # noqa: F401, E402
    animated,
    deep_color,
    graphic,
    pathological,
    photo,
    text,
    transparent,
)
from bench.corpus.synthesis._common import (  # noqa: F401
    get_synth,
    known_kinds,
    register_kind,
)


def synthesize(entry: ManifestEntry) -> Synthesized:
    """Dispatch a manifest entry to its registered synthesizer."""
    fn = get_synth(entry.content_kind)
    return fn(
        seed=entry.seed,
        width=entry.width,
        height=entry.height,
        **entry.params,
    )

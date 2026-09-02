"""TEMPORARY — delete with .github/workflows/verify-dep-bumps.yml.

Generate deterministic AVIF fixtures for the dependency-bump verification.

Written once on the OLD stack so both stacks optimize identical input bytes.
Covers the content types where the libavif quality-scale retune showed the
largest divergence during local investigation: photographic (worst regression),
noise (improved), and flat graphics (dropped to 0% reduction).
"""

import io
import sys
from pathlib import Path

import numpy as np
import pillow_avif  # noqa: F401 — registers the AVIF plugin
from PIL import Image


def _surface(h: int, w: int, noise: float, seed: int) -> np.ndarray:
    """Smooth gradient plus tunable noise — cheap stand-in for photo content."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    base = np.sin(xx / 90) * 0.12 + np.cos(yy / 70) * 0.10 + xx / w * 0.5 + yy / h * 0.3
    return np.clip(base + rng.normal(0, noise, (h, w)), 0, 1)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, noise in (("photo", 0.035), ("noise", 0.15), ("flat", 0.005)):
        f = _surface(576, 768, noise, seed=7)
        arr = np.stack(
            [f * 255, (f**1.15) * 255, np.clip(1 - f * 0.8, 0, 1) * 255], axis=-1
        ).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="AVIF", quality=65, speed=6)
        (out / f"{name}_8bit.avif").write_bytes(buf.getvalue())
        print(f"{name}_8bit.avif: {len(buf.getvalue())} bytes")


if __name__ == "__main__":
    main(sys.argv[1])

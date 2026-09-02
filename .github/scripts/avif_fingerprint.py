"""TEMPORARY — delete with .github/workflows/verify-dep-bumps.yml.

Run every AVIF fixture through the real production optimizer and record a
SHA-256 of each result, so the same fixtures can be compared across the old and
new dependency stacks.

Hashing the optimizer's *output* (not just its size) is the point: a size match
with different bytes would still mean the encoder changed behaviour.
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import PIL  # noqa: E402

from optimizers.avif import AvifOptimizer  # noqa: E402
from schemas import OptimizationConfig  # noqa: E402

# HIGH / MEDIUM / LOW — the benchmark presets from CLAUDE.md.
PRESETS = (40, 60, 75)


def _libavif_version() -> str:
    try:
        from pillow_avif import _avif

        return _avif.libavif_version
    except Exception:  # pragma: no cover - diagnostic only
        return "unknown"


def main(fixture_dir: str, out_path: str) -> None:
    optimizer = AvifOptimizer()
    results = {}

    for path in sorted(Path(fixture_dir).glob("*.avif")):
        data = path.read_bytes()
        for quality in PRESETS:
            result = asyncio.run(optimizer.optimize(data, OptimizationConfig(quality=quality)))
            results[f"{path.name}|q{quality}"] = {
                "sha256": hashlib.sha256(result.optimized_bytes).hexdigest(),
                "size": len(result.optimized_bytes),
                "method": result.method,
            }

    payload = {
        "pillow": PIL.__version__,
        "libavif": _libavif_version(),
        "results": results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"fingerprinted {len(results)} outputs -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

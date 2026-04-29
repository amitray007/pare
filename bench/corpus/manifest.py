"""Manifest schema, loading, and verification.

The manifest is a JSON document that pins the corpus to a specific set of
synthesized inputs. It records pixel-level SHA-256 (raw decoded bytes) as
the canonical determinism contract; encoded byte SHAs are recorded per
platform for diagnostics only, since libjpeg-turbo / libpng outputs vary
across CPU SIMD paths and library builds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import numpy as np
from PIL import Image, ImageSequence

# A synthesizer returns one of:
# - a single Pillow Image (static content);
# - a list of Pillow Images (animated content, frame-by-frame);
# - a numpy uint16 array (10/12-bit deep color, shape (H, W, 3 or 4)).
Synthesized = Union[Image.Image, list[Image.Image], np.ndarray]

MANIFEST_VERSION = 1

# Byte-size buckets. Range is [low, high). xlarge has no upper bound.
BUCKET_RANGES: dict[str, tuple[int, int | None]] = {
    "tiny": (0, 10 * 1024),
    "small": (10 * 1024, 100 * 1024),
    "medium": (100 * 1024, 1024 * 1024),
    "large": (1024 * 1024, 5 * 1024 * 1024),
    "xlarge": (5 * 1024 * 1024, None),
}


class Bucket(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


def bucket_for_size(size_bytes: int) -> Bucket:
    for name, (lo, hi) in BUCKET_RANGES.items():
        if size_bytes >= lo and (hi is None or size_bytes < hi):
            return Bucket(name)
    raise ValueError(f"unbucketable size: {size_bytes}")


@dataclass
class ManifestEntry:
    """A single synthesized corpus item.

    `expected_pixel_sha256` is computed from the synthesizer output's raw
    pixel bytes (after a normalized mode conversion). It is the canonical
    determinism contract — encoded outputs may differ across libraries and
    CPUs, but the pixel array does not.

    `output_formats` lists which encoded files the conversion stage will
    derive from this entry. Bucket validation runs on each derived file.
    """

    name: str
    bucket: Bucket
    content_kind: str
    seed: int
    width: int
    height: int
    output_formats: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    expected_pixel_sha256: str | None = None
    encoded_sha256: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "bucket": self.bucket.value,
            "content_kind": self.content_kind,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "output_formats": list(self.output_formats),
        }
        if self.params:
            d["params"] = self.params
        if self.tags:
            d["tags"] = list(self.tags)
        if self.expected_pixel_sha256:
            d["expected_pixel_sha256"] = self.expected_pixel_sha256
        if self.encoded_sha256:
            d["encoded_sha256"] = self.encoded_sha256
        return d

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ManifestEntry:
        try:
            return cls(
                name=raw["name"],
                bucket=Bucket(raw["bucket"]),
                content_kind=raw["content_kind"],
                seed=int(raw["seed"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
                output_formats=list(raw["output_formats"]),
                params=dict(raw.get("params") or {}),
                tags=list(raw.get("tags") or []),
                expected_pixel_sha256=raw.get("expected_pixel_sha256"),
                encoded_sha256=dict(raw.get("encoded_sha256") or {}),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ManifestSchemaError(f"invalid entry {raw.get('name', '<unnamed>')}: {e}") from e


@dataclass
class Manifest:
    name: str
    library_versions: dict[str, str]
    entries: list[ManifestEntry]
    manifest_version: int = MANIFEST_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "manifest_name": self.name,
            "library_versions": dict(self.library_versions),
            "entries": [e.to_json() for e in self.entries],
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Manifest:
        version = raw.get("manifest_version")
        if version != MANIFEST_VERSION:
            raise ManifestSchemaError(
                f"manifest_version={version!r} not supported; expected {MANIFEST_VERSION}. "
                f"Regenerate with `python -m bench.corpus build`."
            )
        try:
            return cls(
                name=raw["manifest_name"],
                library_versions=dict(raw.get("library_versions") or {}),
                entries=[ManifestEntry.from_json(e) for e in raw["entries"]],
            )
        except (KeyError, TypeError) as e:
            raise ManifestSchemaError(f"invalid manifest header: {e}") from e

    def filter(
        self,
        bucket: str | None = None,
        content_kind: str | None = None,
        tag: str | None = None,
    ) -> list[ManifestEntry]:
        out = list(self.entries)
        if bucket:
            out = [e for e in out if e.bucket.value == bucket]
        if content_kind:
            out = [e for e in out if e.content_kind == content_kind]
        if tag:
            out = [e for e in out if tag in e.tags]
        return out

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        raw = json.loads(Path(path).read_text())
        return cls.from_json(raw)

    def save(self, path: str | Path) -> None:
        atomic_write_json(Path(path), self.to_json())


class ManifestSchemaError(Exception):
    """Manifest JSON did not match the v1 schema."""


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON via tmpfile + rename so partial writes never corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def normalized_mode(image: Image.Image) -> str:
    """Pick a canonical mode for pixel hashing.

    Palette and bilevel modes are expanded so the pixel hash is independent
    of palette ordering. Modes with alpha keep their alpha channel.
    """
    mode = image.mode
    if mode in ("RGB", "RGBA", "L", "LA", "I;16", "I;16L", "I;16B"):
        return mode
    if mode == "P":
        return "RGBA" if "transparency" in image.info else "RGB"
    if mode == "1":
        return "L"
    if mode == "CMYK" or mode == "YCbCr":
        return "RGB"
    return "RGBA" if image.mode.endswith("A") else "RGB"


def pixel_sha256(content: Synthesized) -> str:
    """SHA-256 of raw pixel bytes after mode normalization.

    Accepts three input shapes:

    - Static `Image.Image` — hashes one frame.
    - Animated `Image.Image` (e.g. opened APNG/GIF, `is_animated=True`)
      or `list[Image.Image]` — hashes every frame in order.
    - `numpy.ndarray` — used for 10/12-bit deep color content where
      Pillow's modes can't represent the bit depth. The dtype and shape
      are baked into the digest so a 10-bit and a 16-bit array of the
      same logical pixels never collide.
    """
    h = hashlib.sha256()

    if isinstance(content, np.ndarray):
        h.update(b"ndarray:")
        h.update(str(content.dtype).encode("utf-8"))
        h.update(b":")
        h.update(str(content.shape).encode("utf-8"))
        h.update(b":")
        h.update(content.tobytes())
        return h.hexdigest()

    frames: Iterable[Image.Image]
    if isinstance(content, list):
        frames = content
    elif getattr(content, "is_animated", False):
        frames = ImageSequence.Iterator(content)
    else:
        frames = [content]

    for frame in frames:
        target = normalized_mode(frame)
        if frame.mode != target:
            frame = frame.convert(target)
        h.update(target.encode("utf-8"))
        h.update(b":")
        h.update(f"{frame.size[0]}x{frame.size[1]}".encode("utf-8"))
        h.update(b":")
        h.update(frame.tobytes())
        h.update(b"|")

    return h.hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    mismatches: list[str]
    missing: list[str]
    schema_errors: list[str]

    @property
    def exit_code(self) -> int:
        if self.schema_errors:
            return 3
        if self.missing:
            return 2
        if self.mismatches:
            return 1
        return 0


SynthesizeFunc = Callable[[ManifestEntry], Synthesized]


def verify(manifest: Manifest, synthesize: SynthesizeFunc) -> VerifyResult:
    """Re-synthesize every entry and compare against expected_pixel_sha256.

    Entries with no expected_pixel_sha256 (a freshly-built unsealed manifest)
    are reported as missing. The caller must have run `build --update-manifest`
    first to populate the hashes.
    """
    mismatches: list[str] = []
    missing: list[str] = []
    schema_errors: list[str] = []

    for entry in manifest.entries:
        if entry.expected_pixel_sha256 is None:
            missing.append(f"{entry.name} (manifest has no expected_pixel_sha256)")
            continue
        try:
            image = synthesize(entry)
        except Exception as e:
            schema_errors.append(f"{entry.name}: synthesis failed: {e}")
            continue
        actual = pixel_sha256(image)
        if actual != entry.expected_pixel_sha256:
            mismatches.append(
                f"{entry.name}: expected={entry.expected_pixel_sha256[:12]} "
                f"actual={actual[:12]}"
            )

    return VerifyResult(
        ok=not (mismatches or missing or schema_errors),
        mismatches=mismatches,
        missing=missing,
        schema_errors=schema_errors,
    )


def current_platform_key() -> str:
    return platform.system().lower()


def collect_library_versions() -> dict[str, str]:
    """Snapshot pinned library versions at build time.

    Drift in any of these is a hint that pixel hashes might also drift.
    Recorded into the manifest so a future verify on a different machine
    can warn before failing.
    """
    versions: dict[str, str] = {}
    try:
        from PIL import __version__ as pillow_version

        versions["Pillow"] = pillow_version
    except ImportError:
        pass
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import pillow_heif

        versions["pillow_heif"] = pillow_heif.__version__
    except ImportError:
        pass
    try:
        import jxlpy

        versions["jxlpy"] = getattr(jxlpy, "__version__", "unknown")
    except ImportError:
        pass
    return versions

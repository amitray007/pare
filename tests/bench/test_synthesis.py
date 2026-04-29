"""Synthesizer determinism + registry tests.

Same seed produces identical pixel bytes 3 runs in a row across all
content_kinds. Different seeds produce different bytes (so seed actually
matters, not just a placebo).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bench.corpus.manifest import Bucket, ManifestEntry, pixel_sha256
from bench.corpus.synthesis import known_kinds, synthesize


def _entry(kind: str, *, seed: int = 1, w: int = 96, h: int = 96, **params) -> ManifestEntry:
    if kind == "deep_color_smooth":
        params.setdefault("bit_depth", 10)
    return ManifestEntry(
        name=f"t_{kind}",
        bucket=Bucket.SMALL,
        content_kind=kind,
        seed=seed,
        width=w,
        height=h,
        output_formats=["png"],
        params=params,
    )


@pytest.fixture(params=known_kinds())
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


def test_synth_returns_expected_type(kind: str):
    """Static kinds return Image, animated kinds return list[Image],
    deep-color kinds return uint16 ndarray."""
    out = synthesize(_entry(kind))
    if kind.startswith("animated_"):
        assert isinstance(out, list)
        assert all(isinstance(f, Image.Image) for f in out)
    elif kind.startswith("deep_color_"):
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint16
    else:
        assert isinstance(out, Image.Image)


def test_synth_respects_dimensions(kind: str):
    out = synthesize(_entry(kind, w=80, h=60))
    if kind.startswith("animated_"):
        assert all(f.size == (80, 60) for f in out)
    elif kind.startswith("deep_color_"):
        assert out.shape[:2] == (60, 80)  # (H, W, ...) for ndarrays
    else:
        assert out.size == (80, 60)


def test_synth_is_deterministic_across_three_runs(kind: str):
    a = pixel_sha256(synthesize(_entry(kind, seed=42)))
    b = pixel_sha256(synthesize(_entry(kind, seed=42)))
    c = pixel_sha256(synthesize(_entry(kind, seed=42)))
    assert a == b == c, f"{kind} produced different pixels across runs"


def test_synth_seed_actually_matters(kind: str):
    """Different seeds should yield different outputs — except for kinds
    whose output is seed-independent (e.g. solid fill, fixed patterns)."""
    a = pixel_sha256(synthesize(_entry(kind, seed=1)))
    b = pixel_sha256(synthesize(_entry(kind, seed=2)))

    seed_independent = {
        "path_thin_gradient",
        "path_sharp_diagonal",
        "path_block_aligned_check",
        "path_solid_fill",
        "path_alpha_edge_sprite",
        "path_text_on_flat",
        "path_chroma_clash",
        "deep_color_thin_gradient",
    }
    if kind in seed_independent:
        assert a == b, f"{kind} declared seed-independent but pixels differ"
    else:
        assert a != b, f"{kind} ignored seed parameter"


def test_synthesize_raises_on_unknown_kind():
    bad = ManifestEntry(
        name="bad",
        bucket=Bucket.SMALL,
        content_kind="not_a_real_kind",
        seed=1,
        width=32,
        height=32,
        output_formats=["png"],
    )
    with pytest.raises(ValueError, match="unknown content_kind"):
        synthesize(bad)


def test_known_kinds_includes_all_categories():
    kinds = set(known_kinds())
    expected = {
        "photo_gradient",
        "photo_perlin",
        "photo_noise",
        "graphic_geometric",
        "graphic_palette",
        "text_screenshot",
        "transparent_overlay",
        "transparent_sprite",
        "path_thin_gradient",
        "path_sharp_diagonal",
        "path_block_aligned_check",
        "path_text_on_flat",
        "path_white_noise",
        "path_solid_fill",
        "path_alpha_edge_sprite",
        "path_chroma_clash",
        "animated_translation",
        "animated_fade",
        "animated_sprite_static",
        "animated_redraw",
        "deep_color_smooth",
        "deep_color_thin_gradient",
    }
    assert expected.issubset(kinds), f"missing: {expected - kinds}"


def test_register_kind_rejects_duplicates():
    from bench.corpus.synthesis._common import register_kind

    @register_kind("__test_one_off")
    def _f(**_kw):
        return Image.new("RGB", (1, 1))

    with pytest.raises(ValueError, match="already registered"):

        @register_kind("__test_one_off")
        def _g(**_kw):
            return Image.new("RGB", (1, 1))

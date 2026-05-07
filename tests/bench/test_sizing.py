"""Size-bucket targeting tests."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from bench.corpus.manifest import Bucket
from bench.corpus.sizing import (
    SizingConvergenceError,
    bucket_center,
    fit_bpp,
    in_bucket,
    jpeg_encoder,
    png_encoder,
    refine_to_bucket,
    target_dimensions,
)
from bench.corpus.synthesis import synthesize
from bench.corpus.synthesis._common import get_synth


def test_in_bucket_handles_each_bucket():
    assert in_bucket(0, Bucket.TINY)
    assert in_bucket(5_000, Bucket.TINY)
    assert not in_bucket(20_000, Bucket.TINY)
    assert in_bucket(20_000, Bucket.SMALL)
    assert in_bucket(2_000_000, Bucket.LARGE)
    assert in_bucket(50_000_000, Bucket.XLARGE)


def test_in_bucket_accepts_string_bucket():
    assert in_bucket(5_000, "tiny")


def test_in_bucket_rejects_unknown():
    with pytest.raises(ValueError, match="unknown bucket"):
        in_bucket(0, "humongous")


def test_bucket_center_falls_in_bucket():
    for name in ("tiny", "small", "medium", "large"):
        assert in_bucket(bucket_center(name), name)


def test_png_encoder_produces_valid_png():
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    encoded = png_encoder(img)
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"
    Image.open(io.BytesIO(encoded)).verify()


def test_jpeg_encoder_respects_quality():
    """Use textured content — JPEG of a flat color is dominated by header
    bytes and shows no quality differentiation."""
    textured = synthesize(
        # 256x256 pink noise has plenty of high-frequency content for
        # the quantization tables to actually do something.
        __import__("bench.corpus.manifest", fromlist=["ManifestEntry"]).ManifestEntry(
            name="t",
            bucket=Bucket.SMALL,
            content_kind="photo_noise",
            seed=0,
            width=256,
            height=256,
            output_formats=["jpeg"],
        )
    )
    high_q = jpeg_encoder(quality=95)(textured)
    low_q = jpeg_encoder(quality=20)(textured)
    assert len(high_q) > len(low_q), "lower quality should produce smaller bytes"


def test_fit_bpp_returns_positive_value():
    bpp = fit_bpp(get_synth("photo_perlin"), png_encoder, probe_w=128, probe_h=128)
    assert bpp > 0
    # PNG of pink-noise content should be close to 1-3 bytes per pixel
    assert 0.1 < bpp < 5.0


def test_fit_bpp_rejects_zero_dims():
    with pytest.raises(ValueError, match="positive"):
        fit_bpp(get_synth("photo_perlin"), png_encoder, probe_w=0, probe_h=64)


def test_fit_bpp_rejects_synthesizers_returning_non_image():
    """Animated/deep-color synthesizers don't fit the bpp probe model."""
    with pytest.raises(TypeError):
        fit_bpp(
            get_synth("animated_translation"),
            png_encoder,
            probe_w=64,
            probe_h=64,
        )


def test_target_dimensions_inverse_of_bpp():
    bpp = 1.0
    w, h = target_dimensions(bpp, target_bytes=10_000, aspect=4 / 3)
    # area ≈ target_bytes / bpp = 10_000 px
    assert 8_000 < w * h < 12_000


def test_target_dimensions_respects_aspect():
    w_43, h_43 = target_dimensions(1.0, 10_000, aspect=4 / 3)
    w_11, h_11 = target_dimensions(1.0, 10_000, aspect=1.0)
    assert w_43 / h_43 > w_11 / h_11


def test_target_dimensions_rejects_zero_bpp():
    with pytest.raises(ValueError, match="bpp"):
        target_dimensions(0.0, 1000)


def test_refine_to_bucket_lands_in_small_for_photo_perlin():
    img, w, h, size = refine_to_bucket(
        get_synth("photo_perlin"),
        png_encoder,
        Bucket.SMALL,
    )
    assert in_bucket(size, Bucket.SMALL), f"size={size} dims={w}x{h}"
    assert img.size == (w, h)


def test_refine_to_bucket_lands_in_medium_for_photo_perlin():
    _, _, _, size = refine_to_bucket(
        get_synth("photo_perlin"),
        png_encoder,
        Bucket.MEDIUM,
    )
    assert in_bucket(size, Bucket.MEDIUM)


def test_refine_to_bucket_lands_in_tiny_for_solid_fill():
    """Solid fills compress to nearly nothing — the tiny bucket is the
    natural target. This is the lower-bound case."""
    _, _, _, size = refine_to_bucket(
        get_synth("path_solid_fill"),
        png_encoder,
        Bucket.TINY,
    )
    assert in_bucket(size, Bucket.TINY)


def test_sizing_convergence_error_class_exists():
    """The error class is exported for callers; just verify import path."""
    assert issubclass(SizingConvergenceError, Exception)


def test_refine_to_bucket_works_with_jpeg_encoder():
    _, _, _, size = refine_to_bucket(
        get_synth("photo_perlin"),
        jpeg_encoder(quality=75),
        Bucket.SMALL,
    )
    assert in_bucket(size, Bucket.SMALL)


def test_synthesize_at_target_bucket_dimensions_is_consistent():
    """The image returned by refine_to_bucket should match what
    synthesize() produces at the same dims (no hidden state)."""
    synth = get_synth("photo_perlin")
    img1, w, h, _ = refine_to_bucket(synth, png_encoder, Bucket.SMALL)
    img2 = synth(seed=0, width=w, height=h)
    assert img1.size == img2.size
    # bytes equal because synth is deterministic for fixed seed
    from bench.corpus.manifest import pixel_sha256

    assert pixel_sha256(img1) == pixel_sha256(img2)


def test_synthesize_dispatch_works_with_sizing_dims():
    """End-to-end smoke: pick dims via refine, synth via dispatch table,
    encoded size still in bucket."""
    from bench.corpus.manifest import Bucket as B
    from bench.corpus.manifest import ManifestEntry

    synth = get_synth("photo_perlin")
    _, w, h, _ = refine_to_bucket(synth, png_encoder, B.SMALL)

    entry = ManifestEntry(
        name="probe",
        bucket=B.SMALL,
        content_kind="photo_perlin",
        seed=0,
        width=w,
        height=h,
        output_formats=["png"],
    )
    img = synthesize(entry)
    encoded = png_encoder(img)
    assert in_bucket(len(encoded), B.SMALL)

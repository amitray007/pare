"""Text-heavy content — rendered paragraphs and UI mockups.

Lossless wins in this regime (PNG-8, WebP-LL, JXL-LL). For lossy formats,
this exposes chroma-subsampling artifacts (4:2:0 JPEG eats text on flat
color backgrounds — the canonical failure mode).

Determinism note: Pillow's default font rendering is build-dependent. The
pixel-hash determinism contract will catch any drift on a fresh machine —
the corrective action is to regenerate (and update) the manifest hashes.
A vendored DejaVuSans.ttf at `bench/corpus/fonts/` is a v1 followup.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from bench.corpus.synthesis._common import make_rng, register_kind

_LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute "
    "irure reprehenderit voluptate velit esse cillum fugiat nulla pariatur excepteur "
    "sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit"
).split()


def _default_font(size: int) -> ImageFont.ImageFont:
    """Pillow >= 10 supports a sized default font (Aileron-based)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _gen_paragraph(py_rng, n_words: int) -> str:
    return " ".join(py_rng.choice(_LOREM_WORDS) for _ in range(n_words))


@register_kind("text_screenshot")
def text_screenshot(
    *,
    seed: int,
    width: int,
    height: int,
    bg: tuple[int, int, int] = (250, 250, 252),
    fg: tuple[int, int, int] = (24, 26, 32),
    accent: tuple[int, int, int] = (28, 100, 220),
    title_size: int | None = None,
    body_size: int | None = None,
) -> Image.Image:
    """Document/UI mockup: title, body paragraphs, a colored sidebar accent.

    Picks font sizes proportional to the canvas so the layout stays
    readable across buckets without per-size hand tuning.
    """
    py_rng, _ = make_rng(seed)
    if title_size is None:
        title_size = max(14, height // 18)
    if body_size is None:
        body_size = max(10, height // 36)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    sidebar_w = max(8, width // 32)
    draw.rectangle([0, 0, sidebar_w, height], fill=accent)

    title_font = _default_font(title_size)
    body_font = _default_font(body_size)

    margin = sidebar_w + max(12, width // 40)
    y = max(12, height // 24)

    title = _gen_paragraph(py_rng, 5).title()
    draw.text((margin, y), title, font=title_font, fill=fg)
    y += int(title_size * 1.6)

    line_height = int(body_size * 1.4)
    avail_w = width - margin - max(12, width // 40)

    while y + line_height < height - max(12, height // 24):
        words = _gen_paragraph(py_rng, py_rng.randint(8, 14))
        # crude wrap: cut at a width estimate (Pillow's textlength for the run)
        if hasattr(draw, "textlength"):
            text_w = draw.textlength(words, font=body_font)
        else:
            text_w = body_font.getsize(words)[0]
        scale = max(1.0, text_w / avail_w)
        cut = int(len(words) / scale)
        draw.text((margin, y), words[:cut], font=body_font, fill=fg)
        y += line_height

    return img

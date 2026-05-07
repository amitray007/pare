"""Synthetic SVG generators for the benchmark corpus.

Produces deterministic SVG bytes from (seed, dims) — no Pillow involved.
The synthesizer returns `bytes` (not a PIL Image), which the builder passes
through to the SVG/SVGZ encoders as-is.

Two content kinds:

- `vector_geometric`:  rects, circles, and lines on a coloured background.
  Exercises basic SVG path / attribute optimisation in scour.

- `vector_with_script`:  same shapes *plus* a ``<script>`` block and an
  ``onclick`` event handler on the root ``<svg>`` element.  The SvgOptimizer
  sanitiser must strip both before scour runs — present in the input,
  absent from the output.

Determinism rules
-----------------
- Use ``random.Random(seed)`` only — never the global PRNG.
- No ``datetime`` or ``time`` calls in the generated XML.
- All floating-point coordinates are rounded to 2 decimal places so the
  output is identical regardless of platform float repr.
"""

from __future__ import annotations

import random

from bench.corpus.synthesis._common import register_kind

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    """Convert HSV (all in [0, 1]) to a CSS hex colour string."""
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def _random_colour(rng: random.Random) -> str:
    h = rng.random()
    s = rng.uniform(0.4, 0.9)
    v = rng.uniform(0.4, 0.9)
    return _hsv_to_hex(h, s, v)


def _build_shapes(rng: random.Random, width: int, height: int, n: int) -> list[str]:
    """Generate `n` SVG shape elements as XML strings."""
    elements: list[str] = []
    for _ in range(n):
        kind = rng.choice(("rect", "circle", "line"))
        colour = _random_colour(rng)
        opacity = round(rng.uniform(0.5, 1.0), 2)

        if kind == "rect":
            x = round(rng.uniform(0, width * 0.85), 2)
            y = round(rng.uniform(0, height * 0.85), 2)
            w = round(rng.uniform(width * 0.05, width * 0.4), 2)
            h = round(rng.uniform(height * 0.05, height * 0.4), 2)
            rx = round(rng.uniform(0, min(w, h) * 0.3), 2)
            elements.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"'
                f' fill="{colour}" opacity="{opacity}"/>'
            )
        elif kind == "circle":
            cx = round(rng.uniform(0, width), 2)
            cy = round(rng.uniform(0, height), 2)
            r = round(rng.uniform(width * 0.02, width * 0.2), 2)
            elements.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}"' f' fill="{colour}" opacity="{opacity}"/>'
            )
        else:  # line
            x1 = round(rng.uniform(0, width), 2)
            y1 = round(rng.uniform(0, height), 2)
            x2 = round(rng.uniform(0, width), 2)
            y2 = round(rng.uniform(0, height), 2)
            sw = round(rng.uniform(1.0, max(2.0, width * 0.01)), 2)
            elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
                f' stroke="{colour}" stroke-width="{sw}" opacity="{opacity}"/>'
            )

    return elements


def _svg_document(
    width: int,
    height: int,
    bg_colour: str,
    shapes: list[str],
    *,
    extra_header: str = "",
    extra_body: str = "",
) -> bytes:
    """Assemble a minimal, well-formed SVG document."""
    shape_block = "\n  ".join(shapes)
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}"'
        f"{extra_header}>\n"
        f'  <rect width="{width}" height="{height}" fill="{bg_colour}"/>\n'
        f"  {shape_block}\n"
        f"{extra_body}"
        "</svg>\n"
    )
    return doc.encode("utf-8")


# ---------------------------------------------------------------------------
# Registered synthesizers
# ---------------------------------------------------------------------------


@register_kind("vector_geometric")
def vector_geometric(
    *,
    seed: int,
    width: int,
    height: int,
    n_shapes: int = 40,
) -> bytes:
    """Deterministic SVG: coloured rects, circles, and lines.

    Exercises basic scour optimisation (viewBox normalisation, precision
    reduction, ID shortening).  No scripts or event handlers — safe input.
    """
    rng = random.Random(seed)
    bg = _random_colour(rng)
    shapes = _build_shapes(rng, width, height, n_shapes)
    return _svg_document(width, height, bg, shapes)


@register_kind("vector_with_script")
def vector_with_script(
    *,
    seed: int,
    width: int,
    height: int,
    n_shapes: int = 40,
) -> bytes:
    """Deterministic SVG containing a ``<script>`` block and an event handler.

    The SvgOptimizer sanitiser must strip:
    - the ``<script>`` element entirely
    - the ``onclick`` attribute on the root ``<svg>`` element

    Both should be present in the *input* bytes and absent in the optimised
    output — this exercises the sanitisation code path.
    """
    rng = random.Random(seed)
    bg = _random_colour(rng)
    shapes = _build_shapes(rng, width, height, n_shapes)

    # Inline script block — sanitiser must remove this
    script_block = (
        '  <script type="text/javascript">\n'
        "    // bench-generated script — must be stripped by sanitiser\n"
        "    function benchAlert() { alert('bench'); }\n"
        "  </script>\n"
    )
    # Event handler on root element — sanitiser must remove this attribute
    extra_header = ' onclick="benchAlert()"'

    return _svg_document(
        width,
        height,
        bg,
        shapes,
        extra_header=extra_header,
        extra_body=script_block,
    )

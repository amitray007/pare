"""Image generators for benchmark cases.

Each generator creates a specific content type that exercises different
aspects of the compression pipeline. Uses fixed seeds for reproducibility.
"""

import gzip
import io
import random

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Raster generators
# ---------------------------------------------------------------------------


def photo_like(w: int, h: int, seed: int = 42) -> Image.Image:
    """Smooth gradients + gaussian noise (high entropy, like a camera photo)."""
    img = Image.new("RGB", (w, h))
    rng = random.Random(seed)
    for x in range(w):
        for y in range(h):
            r = int((x / w) * 200 + rng.gauss(0, 25))
            g = int((y / h) * 180 + rng.gauss(0, 25))
            b = int(((x + y) / (w + h)) * 160 + rng.gauss(0, 25))
            img.putpixel((x, y), (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
    return img


def screenshot_like(w: int, h: int, seed: int = 42) -> Image.Image:
    """Flat colors + sharp edges (like a UI screenshot)."""
    img = Image.new("RGB", (w, h), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)
    # Header bar
    draw.rectangle([(0, 0), (w, min(60, h // 4))], fill=(50, 50, 60))
    # Sidebar
    sidebar_w = min(200, w // 4)
    draw.rectangle([(0, min(60, h // 4)), (sidebar_w, h)], fill=(245, 245, 250))
    # Content blocks
    block_count = min(8, max(1, (h - 100) // 40))
    for i in range(block_count):
        y = 80 + i * ((h - 100) // max(1, block_count))
        max_bw = max(w - sidebar_w - 40, sidebar_w + 10)
        bw = rng.randint(sidebar_w, max_bw)
        draw.rectangle(
            [(sidebar_w + 20, y), (sidebar_w + 20 + bw, y + 30)],
            fill=(rng.randint(180, 220),) * 3,
        )
    # Buttons
    if w > 400 and h > 100:
        for i in range(3):
            x = sidebar_w + 20 + i * 140
            if x + 120 < w:
                draw.rounded_rectangle(
                    [(x, h - 70), (x + 120, h - 40)],
                    radius=6,
                    fill=(66, 133, 244),
                )
    return img


def graphic_like(w: int, h: int, seed: int = 42) -> Image.Image:
    """Flat-color shapes (like an illustration or logo)."""
    img = Image.new("RGBA", (w, h), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)
    colors = [
        (66, 133, 244, 255),
        (234, 67, 53, 255),
        (251, 188, 4, 255),
        (52, 168, 83, 255),
        (103, 58, 183, 255),
    ]
    for _ in range(12):
        c = rng.choice(colors)
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        size = rng.randint(w // 10, w // 3)
        if rng.random() > 0.5:
            draw.ellipse([(cx - size, cy - size), (cx + size, cy + size)], fill=c)
        else:
            draw.rectangle([(cx, cy), (cx + size, cy + size)], fill=c)
    return img


def gradient(w: int, h: int) -> Image.Image:
    """Pure smooth gradient."""
    img = Image.new("RGB", (w, h))
    for x in range(w):
        for y in range(h):
            img.putpixel((x, y), (int(x * 255 / w), int(y * 255 / h), 128))
    return img


def solid(w: int, h: int) -> Image.Image:
    """Single solid color."""
    return Image.new("RGB", (w, h), color=(100, 150, 200))


def transparent_png(w: int, h: int, seed: int = 42) -> Image.Image:
    """RGBA with partial transparency."""
    img = Image.new("RGBA", (w, h), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)
    for _ in range(5):
        x, y = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(20, w // 3)
        alpha = rng.randint(80, 200)
        draw.ellipse(
            [(x - r, y - r), (x + r, y + r)],
            fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), alpha),
        )
    return img


def palette_png(w: int, h: int, seed: int = 42) -> Image.Image:
    """Palette-mode PNG (already indexed colors)."""
    return graphic_like(w, h, seed).convert("P", palette=Image.ADAPTIVE, colors=64)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def encode_image(
    img: Image.Image,
    fmt: str,
    quality: int = 95,
    compression: str | None = None,
) -> bytes:
    """Encode a PIL Image to bytes in the given format."""
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
    elif fmt == "jpeg":
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    elif fmt == "webp":
        img.convert("RGB").save(buf, format="WEBP", quality=quality)
    elif fmt == "gif":
        img.convert("P", palette=Image.ADAPTIVE, colors=256).save(buf, format="GIF")
    elif fmt == "bmp":
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        img.save(buf, format="BMP")
    elif fmt == "tiff":
        save_kwargs = {"format": "TIFF"}
        if compression:
            save_kwargs["compression"] = compression
        img.convert("RGB").save(buf, **save_kwargs)
    elif fmt == "avif":
        import pillow_avif  # noqa: F401

        img.convert("RGB").save(buf, format="AVIF", quality=quality)
    elif fmt == "heif":
        import pillow_heif  # noqa: F811

        pillow_heif.register_heif_opener()
        img.convert("RGB").save(buf, format="HEIF", quality=quality)
    elif fmt == "jxl":
        import jxlpy  # noqa: F401 — registers JXL plugin with Pillow

        img.convert("RGB").save(buf, format="JXL", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SVG generators
# ---------------------------------------------------------------------------


def svg_simple(w: int = 800, h: int = 600) -> bytes:
    return (
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="#f0f0f0"/>
  <circle cx="{w // 2}" cy="{h // 2}" r="{min(w, h) // 3}" fill="#4a90d9"/>
</svg>""".encode()
    )


def svg_complex(w: int = 800, h: int = 600, seed: int = 42) -> bytes:
    rng = random.Random(seed)
    paths = []
    for _ in range(30):
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        x2, y2 = rng.randint(0, w), rng.randint(0, h)
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        color = f"#{rng.randint(0, 0xFFFFFF):06x}"
        paths.append(
            f'  <path d="M{x1},{y1} Q{cx},{cy} {x2},{y2}" '
            f'stroke="{color}" fill="none" stroke-width="2"/>'
        )
    return (
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <metadata><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""><dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Complex</dc:title></rdf:Description></rdf:RDF></metadata>
  <title>Test</title>
  <desc>A complex SVG for benchmarking</desc>
  <rect width="{w}" height="{h}" fill="#ffffff"/>
{chr(10).join(paths)}
</svg>""".encode()
    )


def svg_bloated(w: int = 800, h: int = 600, seed: int = 42) -> bytes:
    rng = random.Random(seed)
    elements = []
    for i in range(50):
        x, y = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(5, 40)
        color = f"#{rng.randint(0, 0xFFFFFF):06x}"
        elements.append(
            f"  <!-- Element number {i} -->\n"
            f'  <circle id="very-long-identifier-for-circle-number-{i}" '
            f'cx="{x}" cy="{y}" r="{r}" fill="{color}" '
            f'stroke="none" stroke-width="0" opacity="1.0" />'
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by Test Framework -->
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>Bloated SVG</dc:title>
        <dc:creator>Benchmark</dc:creator>
      </rdf:Description>
    </rdf:RDF>
  </metadata>
  <title>Bloated SVG</title>
  <desc>SVG with metadata, comments, and long IDs</desc>
  <defs><style type="text/css">.unused {{ fill: red; }}</style></defs>
{chr(10).join(elements)}
</svg>""".encode()


def svgz_from_svg(svg_bytes: bytes) -> bytes:
    return gzip.compress(svg_bytes, compresslevel=6)

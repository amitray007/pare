from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.subprocess_runner import run_tool

# Total pixel-frames above which palette quantization (--colors=N) becomes a
# no-op on chart/diagram-style content but still costs per-frame CPU.
# Calibrated against iter3 bench data: wdqs_barchart (341 M pixel-frames, Δ=0)
# vs rotating_earth (7 M) and ipcc_climate (1.3 M) which both benefit.
_LARGE_ANIM_PIXEL_FRAMES = 100_000_000


def _count_gif_pixel_frames(data: bytes) -> tuple[int, int, int]:
    """Return (width, height, frame_count) by walking GIF block structure.

    Parses only the GIF header and block types — no pixel data decoded.
    Returns (0, 0, 0) on any parse error so the caller keeps the full flag set.

    GIF89a block layout (just enough to count Image Descriptors / frames):
      Bytes 0-5:   signature (GIF87a / GIF89a)
      Bytes 6-7:   logical screen width  (LE uint16)
      Bytes 8-9:   logical screen height (LE uint16)
      Byte  10:    packed — bit7 = global color table present, bits0-2 = GCT exp
      Bytes 11-12: background index, pixel aspect ratio  (skipped)
      [Optional] global color table: 3 × 2^(GCT_exp+1) bytes
      Then zero or more blocks until trailer (0x3B):
        0x21 — Extension: 1-byte label, then sub-block chain
        0x2C — Image Descriptor (= 1 frame): 9-byte header, optional LCT,
                LZW min code size byte, then sub-block chain
        0x3B — Trailer: end of data
    Sub-block chain: each block is 1-byte count + count bytes; chain ends at count=0.
    """
    try:
        n = len(data)
        if n < 13 or data[:6] not in (b"GIF87a", b"GIF89a"):
            return (0, 0, 0)

        width = data[6] | (data[7] << 8)
        height = data[8] | (data[9] << 8)
        packed = data[10]

        pos = 13  # skip signature(6) + w(2) + h(2) + packed(1) + bg(1) + aspect(1)

        # Skip global color table if present
        if packed & 0x80:
            gct_size = (packed & 0x07) + 1  # 2^(exp+1) entries, 3 bytes each
            pos += 3 * (1 << gct_size)

        frame_count = 0

        while pos < n:
            block_type = data[pos]
            pos += 1

            if block_type == 0x3B:  # Trailer
                break

            elif block_type == 0x21:  # Extension block
                if pos >= n:
                    break
                pos += 1  # skip extension label byte
                # Skip sub-block chain
                while pos < n:
                    sub_len = data[pos]
                    pos += 1
                    if sub_len == 0:
                        break
                    pos += sub_len

            elif block_type == 0x2C:  # Image Descriptor → one frame
                frame_count += 1
                # 9-byte descriptor: left(2) top(2) w(2) h(2) packed(1)
                if pos + 9 > n:
                    break
                local_packed = data[pos + 8]
                pos += 9
                # Skip local color table if present
                if local_packed & 0x80:
                    lct_size = (local_packed & 0x07) + 1
                    pos += 3 * (1 << lct_size)
                # Skip LZW minimum code size byte
                if pos >= n:
                    break
                pos += 1
                # Skip LZW data sub-block chain
                while pos < n:
                    sub_len = data[pos]
                    pos += 1
                    if sub_len == 0:
                        break
                    pos += sub_len

            else:
                # Unrecognised block type — malformed GIF; stop walking
                break

        return (width, height, frame_count)

    except Exception:
        return (0, 0, 0)


class GifOptimizer(BaseOptimizer):
    """GIF optimization via gifsicle.

    Pipeline varies by quality setting:
    - quality < 50:  --optimize=3 --lossy=80 --colors 128 (aggressive)
    - quality < 70:  --optimize=3 --lossy=30 --colors 192 (moderate)
    - quality >= 70: --optimize=3 (lossless only)

    gifsicle --optimize=3 performs lossless optimization (frame bbox
    shrinking, disposal method optimization, LZW recompression).
    --lossy enables lossy LZW compression that modifies pixel values
    slightly for better compression ratios.
    --colors reduces the palette size for better LZW compression.

    Large-animation guard: for animated GIFs above _LARGE_ANIM_PIXEL_FRAMES
    total pixel-frames, --colors=N is dropped from lossy presets.  On chart/
    diagram-style content the palette is already minimal so quantization adds
    zero compression benefit but costs per-frame CPU (iter3 bench: +4500 ms,
    0% additional reduction on wdqs_barchart.gif with 736 frames).
    """

    format = ImageFormat.GIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        cmd = ["gifsicle", "--optimize=3"]

        if config.quality < 50:
            cmd.append("--lossy=80")
            # Check whether palette quantization would be a no-op on a large animation
            w, h, frames = _count_gif_pixel_frames(data)
            if (w * h * frames) > _LARGE_ANIM_PIXEL_FRAMES:
                method = "gifsicle --lossy=80 (palette skipped: large animation)"
            else:
                cmd.extend(["--colors", "128"])
                method = "gifsicle --lossy=80 --colors=128"
        elif config.quality < 70:
            cmd.append("--lossy=30")
            w, h, frames = _count_gif_pixel_frames(data)
            if (w * h * frames) > _LARGE_ANIM_PIXEL_FRAMES:
                method = "gifsicle --lossy=30 (palette skipped: large animation)"
            else:
                cmd.extend(["--colors", "192"])
                method = "gifsicle --lossy=30 --colors=192"
        else:
            method = "gifsicle"

        stdout, stderr, rc = await run_tool(cmd, data)
        return self._build_result(data, stdout, method)

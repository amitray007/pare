from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import detect_format, ImageFormat
from optimizers.png import PngOptimizer
from optimizers.jpeg import JpegOptimizer
from optimizers.webp import WebpOptimizer
from optimizers.gif import GifOptimizer
from optimizers.svg import SvgOptimizer
from optimizers.avif import AvifOptimizer
from optimizers.heic import HeicOptimizer
from optimizers.passthrough import PassthroughOptimizer


# Optimizer registry — initialized once at import time
OPTIMIZERS = {
    ImageFormat.PNG: PngOptimizer(),
    ImageFormat.APNG: PngOptimizer(),
    ImageFormat.JPEG: JpegOptimizer(),
    ImageFormat.WEBP: WebpOptimizer(),
    ImageFormat.GIF: GifOptimizer(),
    ImageFormat.SVG: SvgOptimizer(),
    ImageFormat.SVGZ: SvgOptimizer(),
    ImageFormat.AVIF: AvifOptimizer(),
    ImageFormat.HEIC: HeicOptimizer(),
    ImageFormat.TIFF: PassthroughOptimizer(ImageFormat.TIFF),
    ImageFormat.BMP: PassthroughOptimizer(ImageFormat.BMP),
    ImageFormat.PSD: PassthroughOptimizer(ImageFormat.PSD),
}


async def optimize_image(
    data: bytes,
    config: OptimizationConfig,
) -> OptimizeResult:
    """Detect format and dispatch to the correct optimizer.

    Args:
        data: Raw image bytes.
        config: Optimization parameters.

    Returns:
        OptimizeResult with optimized bytes and stats.

    Raises:
        UnsupportedFormatError: If format is not recognized.
    """
    fmt = detect_format(data)
    optimizer = OPTIMIZERS[fmt]
    return await optimizer.optimize(data, config)

from optimizers.avif import AvifOptimizer
from optimizers.bmp import BmpOptimizer
from optimizers.gif import GifOptimizer
from optimizers.heic import HeicOptimizer
from optimizers.jpeg import JpegOptimizer
from optimizers.jxl import JxlOptimizer
from optimizers.png import PngOptimizer
from optimizers.svg import SvgOptimizer
from optimizers.tiff import TiffOptimizer
from optimizers.webp import WebpOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat, detect_format

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
    ImageFormat.TIFF: TiffOptimizer(),
    ImageFormat.BMP: BmpOptimizer(),
    ImageFormat.JXL: JxlOptimizer(),
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

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.subprocess_runner import run_tool


class GifOptimizer(BaseOptimizer):
    """GIF optimization via gifsicle.

    gifsicle --optimize=3 performs:
    - Shrinks frame bounding boxes
    - Optimizes frame disposal methods
    - Re-compresses LZW per frame

    Quality parameter is ignored — gifsicle is lossless optimization only.
    Handles both static and animated GIFs transparently.
    """

    format = ImageFormat.GIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        stdout, stderr, rc = await run_tool(
            ["gifsicle", "--optimize=3"],
            data,
        )
        return self._build_result(data, stdout, "gifsicle")

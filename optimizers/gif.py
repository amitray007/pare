from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat
from utils.subprocess_runner import run_tool


class GifOptimizer(BaseOptimizer):
    """GIF optimization via gifsicle.

    Pipeline varies by quality setting:
    - quality < 50:  --optimize=3 --lossy=80 (aggressive lossy)
    - quality < 70:  --optimize=3 --lossy=30 (moderate lossy)
    - quality >= 70: --optimize=3 (lossless only)

    gifsicle --optimize=3 performs lossless optimization (frame bbox
    shrinking, disposal method optimization, LZW recompression).
    --lossy enables lossy LZW compression that modifies pixel values
    slightly for better compression ratios.
    """

    format = ImageFormat.GIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        cmd = ["gifsicle", "--optimize=3"]

        if config.quality < 50:
            cmd.append("--lossy=80")
            method = "gifsicle --lossy=80"
        elif config.quality < 70:
            cmd.append("--lossy=30")
            method = "gifsicle --lossy=30"
        else:
            method = "gifsicle"

        stdout, stderr, rc = await run_tool(cmd, data)
        return self._build_result(data, stdout, method)

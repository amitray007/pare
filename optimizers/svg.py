import gzip

from scour.scour import parse_args as scour_parse_args
from scour.scour import scourString

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from security.svg_sanitizer import sanitize_svg
from utils.format_detect import ImageFormat


class SvgOptimizer(BaseOptimizer):
    """SVG/SVGZ optimization via scour (in-process Python library).

    Pipeline varies by quality setting:
    - quality < 50:  Aggressive — strip metadata, precision=3, shorten IDs
    - quality < 70:  Moderate  — strip metadata, precision=5, shorten IDs
    - quality >= 70: Gentle    — keep metadata, no precision reduction

    SVG:  Input → sanitize → scour → Output
    SVGZ: Input → gunzip → sanitize → scour → gzip → Output
    """

    format = ImageFormat.SVG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        is_svgz = data[:2] == b"\x1f\x8b"

        if is_svgz:
            svg_bytes = gzip.decompress(data)
        else:
            svg_bytes = data

        # Sanitize SVG before optimization (strip scripts, event handlers, XXE)
        svg_bytes = sanitize_svg(svg_bytes)
        svg_text = svg_bytes.decode("utf-8", errors="replace")

        optimized_text = self._run_scour(svg_text, config)
        optimized_bytes = optimized_text.encode("utf-8")

        if is_svgz:
            optimized_bytes = gzip.compress(optimized_bytes, compresslevel=9)
            fmt = ImageFormat.SVGZ
        else:
            fmt = ImageFormat.SVG

        self.format = fmt
        return self._build_result(data, optimized_bytes, "scour")

    def _run_scour(self, svg_text: str, config: OptimizationConfig) -> str:
        """Run scour with quality-dependent options.

        Higher quality (>= 70): minimal changes, preserve metadata.
        Lower quality (< 50): aggressive — strip everything, reduce precision.
        """
        args = [
            "--enable-viewboxing",
            "--indent=none",
        ]

        if config.strip_metadata:
            args += [
                "--remove-metadata",
                "--strip-xml-prolog",
                "--remove-descriptive-elements",
                "--enable-comment-stripping",
                "--shorten-ids",
            ]

        # Coordinate precision reduction for aggressive presets
        if config.quality < 50:
            args.append("--set-precision=3")
        elif config.quality < 70:
            args.append("--set-precision=5")

        scour_options = scour_parse_args(args)
        return scourString(svg_text, options=scour_options)

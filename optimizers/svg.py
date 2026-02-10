import gzip

from scour.scour import scourString, parse_args as scour_parse_args

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from security.svg_sanitizer import sanitize_svg
from utils.format_detect import ImageFormat


class SvgOptimizer(BaseOptimizer):
    """SVG/SVGZ optimization via scour (in-process Python library).

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

        optimized_text = self._run_scour(svg_text)
        optimized_bytes = optimized_text.encode("utf-8")

        if is_svgz:
            optimized_bytes = gzip.compress(optimized_bytes, compresslevel=9)
            fmt = ImageFormat.SVGZ
        else:
            fmt = ImageFormat.SVG

        self.format = fmt
        return self._build_result(data, optimized_bytes, "scour")

    def _run_scour(self, svg_text: str) -> str:
        """Run scour in-process.

        Scour options:
        - remove-metadata: Strip editor metadata
        - enable-viewboxing: Add viewBox if missing
        - strip-xml-prolog: Remove <?xml?> declaration
        - remove-descriptive-elements: Remove <title>, <desc>, <metadata>
        - enable-comment-stripping: Remove XML comments
        - shorten-ids: Shorten element IDs
        - indent=none: Remove indentation
        """
        scour_options = scour_parse_args([
            "--remove-metadata",
            "--enable-viewboxing",
            "--strip-xml-prolog",
            "--remove-descriptive-elements",
            "--enable-comment-stripping",
            "--shorten-ids",
            "--indent=none",
        ])

        return scourString(svg_text, options=scour_options)

from abc import ABC, abstractmethod

from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class BaseOptimizer(ABC):
    """Abstract base for format-specific optimizers."""

    format: ImageFormat

    @abstractmethod
    async def optimize(
        self,
        data: bytes,
        config: OptimizationConfig,
    ) -> OptimizeResult:
        """Optimize image bytes.

        Args:
            data: Raw input image bytes.
            config: Optimization parameters from the request.

        Returns:
            OptimizeResult with optimized bytes and stats.
        """

    def _build_result(
        self,
        original: bytes,
        optimized: bytes,
        method: str,
    ) -> OptimizeResult:
        """Build a result, enforcing the optimization guarantee.

        If optimized >= original, returns original with 0% reduction
        and method="none". The API never returns a file larger than input.
        """
        original_size = len(original)
        optimized_size = len(optimized)

        if optimized_size >= original_size:
            return OptimizeResult(
                success=True,
                original_size=original_size,
                optimized_size=original_size,
                reduction_percent=0.0,
                format=self.format.value,
                method="none",
                optimized_bytes=original,
                message="Image is already optimized",
            )

        reduction = round((1 - optimized_size / original_size) * 100, 1)
        return OptimizeResult(
            success=True,
            original_size=original_size,
            optimized_size=optimized_size,
            reduction_percent=reduction,
            format=self.format.value,
            method=method,
            optimized_bytes=optimized,
        )

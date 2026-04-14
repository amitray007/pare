import asyncio

from config import settings
from exceptions import BackpressureError

# Format-specific memory multipliers (compressed_size -> estimated peak memory).
# Derived from audit: accounts for decompressed pixels + intermediate buffers.
# Post-optimization values (after sequential execution for large images).
MEMORY_MULTIPLIERS: dict[str, int] = {
    "tiff": 6,
    "png": 5,
    "jpeg": 4,
    "webp": 4,
    "avif": 4,
    "heic": 4,
    "jxl": 4,
    "bmp": 3,
    "gif": 2,
    "svg": 1,
    "svgz": 1,
    "apng": 5,
}


class CompressionGate:
    """Controls concurrent compression jobs with backpressure.

    - Semaphore limits active jobs (configurable, defaults to CPU count)
    - Queue depth limit prevents OOM from queued payloads
    - Memory budget tracking rejects requests that would exceed the budget
    - When queue or memory budget is full, returns 503 immediately with Retry-After
    """

    def __init__(
        self,
        semaphore_size: int | None = None,
        max_queue: int | None = None,
        memory_budget_bytes: int | None = None,
    ):
        size = semaphore_size if semaphore_size is not None else settings.compression_semaphore_size
        self._semaphore = asyncio.Semaphore(size)
        self._semaphore_size = size
        self._queue_depth = 0
        self._max_queue = max_queue if max_queue is not None else settings.max_queue_depth
        self._lock = asyncio.Lock()
        self._memory_used = 0
        budget = memory_budget_bytes if memory_budget_bytes is not None else 0
        if budget <= 0:
            budget = settings.memory_budget_mb * 1024 * 1024
        self._memory_budget = budget

    async def acquire(self, estimated_memory: int = 0):
        """Acquire a compression slot.

        Args:
            estimated_memory: Estimated peak memory for this request (bytes).
                If 0, only slot-based backpressure applies.

        Raises BackpressureError (503) if queue is full or memory budget exceeded.
        """
        async with self._lock:
            if self._queue_depth >= self._max_queue:
                raise BackpressureError(
                    "Compression queue full. Try again shortly.",
                    retry_after=5,
                )
            # Intentional: `self._memory_used > 0` ensures the first request is always
            # admitted even if its estimate exceeds the budget. Without this, a single
            # large file could never be processed if its estimate > budget. The budget
            # protects against concurrent accumulation, not single-request size.
            if (
                estimated_memory > 0
                and self._memory_used > 0
                and self._memory_used + estimated_memory > self._memory_budget
            ):
                raise BackpressureError(
                    "Memory budget exceeded. Try again shortly.",
                    retry_after=5,
                )
            self._queue_depth += 1
            self._memory_used += estimated_memory

        try:
            await self._semaphore.acquire()
        except BaseException:
            # Roll back tracking if semaphore acquire is cancelled (e.g., client disconnect)
            async with self._lock:
                self._queue_depth = max(0, self._queue_depth - 1)
                self._memory_used = max(0, self._memory_used - estimated_memory)
            raise

    def release(self, estimated_memory: int = 0):
        """Release a compression slot."""
        self._semaphore.release()
        self._queue_depth = max(0, self._queue_depth - 1)
        self._memory_used = max(0, self._memory_used - estimated_memory)

    @property
    def active_jobs(self) -> int:
        return self._semaphore_size - self._semaphore._value

    @property
    def queued_jobs(self) -> int:
        return max(0, self._queue_depth - self.active_jobs)


# Module-level singletons
compression_gate = CompressionGate()
estimate_gate = CompressionGate(
    semaphore_size=settings.estimate_semaphore_size,
    max_queue=settings.estimate_queue_depth,
)

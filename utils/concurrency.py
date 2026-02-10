import asyncio

from config import settings
from exceptions import BackpressureError


class CompressionGate:
    """Controls concurrent compression jobs with backpressure.

    - Semaphore limits active jobs to CPU count (configurable)
    - Queue depth limit prevents OOM from queued payloads (up to 32MB each)
    - When queue is full, returns 503 immediately with Retry-After
    """

    def __init__(self):
        self._semaphore = asyncio.Semaphore(settings.compression_semaphore_size)
        self._queue_depth = 0
        self._max_queue = settings.max_queue_depth
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a compression slot.

        Raises BackpressureError (503) if queue is full.
        """
        async with self._lock:
            if self._queue_depth >= self._max_queue:
                raise BackpressureError(
                    "Compression queue full. Try again shortly.",
                    retry_after=5,
                )
            self._queue_depth += 1

        await self._semaphore.acquire()

    def release(self):
        """Release a compression slot."""
        self._semaphore.release()
        self._queue_depth -= 1

    @property
    def active_jobs(self) -> int:
        return settings.compression_semaphore_size - self._semaphore._value

    @property
    def queued_jobs(self) -> int:
        return max(0, self._queue_depth - self.active_jobs)


# Module-level singleton
compression_gate = CompressionGate()

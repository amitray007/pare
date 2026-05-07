"""Load mode: concurrent-request throughput and backpressure test.

For each case, launches N concurrent optimize_image() calls through a
freshly-constructed CompressionGate (NOT the production singleton).
Measures throughput, 503 rate, latency tail, and queue wait under
controlled contention.

At high N relative to semaphore_size + queue_depth, many requests will
hit BackpressureError — this is expected and is the whole point of the
test. The 503 rate tells you how much headroom your CompressionGate
config provides under a given concurrency level.

Per-case output schema (superset of quick mode):

    {
      "case_id": "...",
      "name": "...",
      "bucket": "small",
      "format": "jpeg",
      "preset": "high",
      "input_size": 88644,
      "iteration": 0,
      "measurement": { ... },          # covers the whole load window; peak_rss_kb is
                                       # the watermark across all concurrent requests
      "tool_invocations": [...],       # accumulated wall across successful sub-requests
      "reduction_pct": 71.3,           # mean across successful requests
      "method": "jpegli",              # most-common method observed
      "optimized_size": 25476,         # mean output size across successes
      "load": {
        "n_concurrent": 30,
        "semaphore_size": 8,
        "queue_depth": 16,
        "n_success": 22,
        "n_503": 8,
        "n_503_queue": 8,
        "n_503_memory": 0,
        "n_error": 0,
        "wall_ms": 1240.5,
        "throughput_per_sec": 17.74,
        "ok_rate": 0.733,
        "request_latency_ms": {        # wall from acquire-start to release for SUCCESSES
          "p50": 230, "p95": 870, "p99": 1100, "max": 1140
        },
        "queue_wait_ms": {             # wall from acquire-start to acquire-grant for SUCCESSES
          "p50": 50, "p95": 320, "p99": 480, "max": 510
        },
        "gate_observed": {             # sampled at ~10 ms intervals during the run
          "max_active_jobs": 8,
          "max_queued_jobs": 16
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import Any

from bench.runner.case import Case
from bench.runner.measure import measure
from bench.runner.modes.quick import measurement_to_dict
from bench.runner.probe import collect_tool_invocations
from bench.runner.stats import percentile
from exceptions import BackpressureError
from optimizers.router import optimize_image
from schemas import OptimizationConfig
from utils.concurrency import MEMORY_MULTIPLIERS, CompressionGate

logger = logging.getLogger(__name__)

# Gate-state sampling interval in seconds.
_SAMPLER_INTERVAL_S = 0.010  # 10 ms


async def _gate_sampler(
    gate: CompressionGate,
    stop_event: asyncio.Event,
    max_active: list[int],
    max_queued: list[int],
) -> None:
    """Poll gate.active_jobs / gate.queued_jobs every ~10 ms until stopped.

    Uses asyncio.sleep so it's cancellation-safe. When the task is
    cancelled (after all requests complete) the CancelledError propagates
    normally; we do a final sample before exiting so we don't miss the peak.
    """
    try:
        while not stop_event.is_set():
            active = gate.active_jobs
            queued = gate.queued_jobs
            if active > max_active[0]:
                max_active[0] = active
            if queued > max_queued[0]:
                max_queued[0] = queued
            await asyncio.sleep(_SAMPLER_INTERVAL_S)
    except asyncio.CancelledError:
        # Final sample on cancellation to catch any last-moment peak.
        active = gate.active_jobs
        queued = gate.queued_jobs
        if active > max_active[0]:
            max_active[0] = active
        if queued > max_queued[0]:
            max_queued[0] = queued
        raise


async def _one_request(
    gate: CompressionGate,
    data: bytes,
    config: OptimizationConfig,
    fmt: str,
) -> dict[str, Any]:
    """Execute one optimize request through the given gate.

    Returns a dict capturing timing, status, and optimization result.
    Mirrors the production try/finally pattern from routers/optimize.py.
    """
    estimated = len(data) * MEMORY_MULTIPLIERS.get(fmt, 4)
    acquire_start = time.perf_counter_ns()

    # --- Acquire gate slot ---
    try:
        try:
            await gate.acquire(estimated_memory=estimated)
        except BackpressureError as exc:
            # Distinguish queue-depth rejection from memory-budget rejection by
            # message substring — BackpressureError is a single class; the gate
            # only sets attribution in the message string (not a subclass).
            # "Compression queue full." → queue-depth cap hit
            # "Memory budget exceeded." → memory budget cap hit
            msg = str(exc)
            status_503 = "503_memory" if "Memory budget exceeded" in msg else "503_queue"
            return {
                "status": status_503,
                "acquire_start_ns": acquire_start,
                "acquire_grant_ns": None,
                "release_ns": time.perf_counter_ns(),
                "result": None,
            }

        acquire_grant = time.perf_counter_ns()

        # --- Run optimizer (release in finally) ---
        try:
            with collect_tool_invocations() as invocations:
                result = await optimize_image(data, config)
            return {
                "status": "success",
                "acquire_start_ns": acquire_start,
                "acquire_grant_ns": acquire_grant,
                "release_ns": time.perf_counter_ns(),
                "result": result,
                "invocations": list(invocations),
            }
        finally:
            gate.release(estimated_memory=estimated)

    except BackpressureError as exc:
        # Should not reach here (caught above), but guard defensively.
        msg = str(exc)
        status_503 = "503_memory" if "Memory budget exceeded" in msg else "503_queue"
        return {
            "status": status_503,
            "acquire_start_ns": acquire_start,
            "acquire_grant_ns": None,
            "release_ns": time.perf_counter_ns(),
            "result": None,
        }
    except Exception as exc:
        # Non-BackpressureError failure (UnsupportedFormatError, optimizer crash, etc.)
        return {
            "status": "error",
            "acquire_start_ns": acquire_start,
            "acquire_grant_ns": None,
            "release_ns": time.perf_counter_ns(),
            "result": None,
            "error_message": f"{type(exc).__name__}: {exc}",
        }


async def _run_one_load_case(
    case: Case,
    *,
    n_concurrent: int,
    semaphore_size: int,
    queue_depth: int,
    memory_budget_mb: int = 0,
) -> dict[str, Any]:
    """Run N concurrent optimize requests for one case through a fresh gate."""
    input_data = case.load()
    config = OptimizationConfig(quality=case.quality)

    # Fresh gate per case — do NOT share state across cases or use the singleton.
    memory_budget_bytes = memory_budget_mb * 1024 * 1024 if memory_budget_mb > 0 else None
    gate = CompressionGate(
        semaphore_size=semaphore_size,
        max_queue=queue_depth,
        memory_budget_bytes=memory_budget_bytes,
    )

    # Gate-state sampler tracking mutable max values via single-element lists.
    max_active: list[int] = [0]
    max_queued: list[int] = [0]
    stop_event = asyncio.Event()

    base: dict[str, Any] = {
        "case_id": case.case_id,
        "name": case.name,
        "bucket": case.bucket,
        "format": case.fmt,
        "preset": case.preset,
        "input_size": case.input_size,
        "iteration": 0,
    }

    # Wrap the entire concurrent launch window in measure() so peak_rss_kb
    # captures the watermark across all concurrent sub-requests.
    with measure() as m:
        sampler_task = asyncio.create_task(_gate_sampler(gate, stop_event, max_active, max_queued))

        wall_start = time.perf_counter_ns()

        tasks = [
            asyncio.create_task(_one_request(gate, input_data, config, case.fmt))
            for _ in range(n_concurrent)
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        wall_end = time.perf_counter_ns()

        # Stop sampler cleanly.
        stop_event.set()
        # One final sample before cancelling, to avoid missing the peak.
        max_active[0] = max(max_active[0], gate.active_jobs)
        max_queued[0] = max(max_queued[0], gate.queued_jobs)
        sampler_task.cancel()
        await asyncio.gather(sampler_task, return_exceptions=True)

    # --- Aggregate outcomes ---
    n_success = 0
    n_503_queue = 0
    n_503_memory = 0
    n_error = 0

    latency_ms_list: list[float] = []  # release - acquire_start (successes only)
    queue_wait_ms_list: list[float] = []  # acquire_grant - acquire_start (successes only)

    reduction_pcts: list[float] = []
    optimized_sizes: list[int] = []
    methods: list[str] = []
    all_invocations: list[dict[str, Any]] = []

    for outcome in outcomes:
        # asyncio.gather(return_exceptions=True) can return exceptions directly
        # if a task raised outside our try/except — treat those as errors.
        if isinstance(outcome, BaseException):
            n_error += 1
            continue

        status = outcome.get("status", "error")
        if status == "success":
            n_success += 1
            start_ns = outcome["acquire_start_ns"]
            grant_ns = outcome["acquire_grant_ns"]
            release_ns = outcome["release_ns"]
            latency_ms_list.append((release_ns - start_ns) / 1e6)
            queue_wait_ms_list.append((grant_ns - start_ns) / 1e6)

            result = outcome["result"]
            reduction_pcts.append(result.reduction_percent)
            optimized_sizes.append(result.optimized_size)
            methods.append(result.method or "")
            for inv in outcome.get("invocations", []):
                all_invocations.append(
                    {"tool": inv.tool, "wall_ms": inv.wall_ms, "exit_code": inv.exit_code}
                )
        elif status == "503_queue":
            n_503_queue += 1
        elif status == "503_memory":
            n_503_memory += 1
        else:
            n_error += 1

    # Per-latency percentiles
    def _pcts(data: list[float]) -> dict[str, float]:
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "p50": round(percentile(data, 50), 1),
            "p95": round(percentile(data, 95), 1),
            "p99": round(percentile(data, 99), 1),
            "max": round(max(data), 1),
        }

    wall_ms = (wall_end - wall_start) / 1e6
    throughput_per_sec = (n_success / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0
    ok_rate = n_success / n_concurrent if n_concurrent > 0 else 0.0

    n_503 = n_503_queue + n_503_memory

    load_block: dict[str, Any] = {
        "n_concurrent": n_concurrent,
        "semaphore_size": semaphore_size,
        "queue_depth": queue_depth,
        "n_success": n_success,
        "n_503": n_503,
        "n_503_queue": n_503_queue,
        "n_503_memory": n_503_memory,
        "n_error": n_error,
        "wall_ms": round(wall_ms, 1),
        "throughput_per_sec": round(throughput_per_sec, 2),
        "ok_rate": round(ok_rate, 3),
        "request_latency_ms": _pcts(latency_ms_list),
        "queue_wait_ms": _pcts(queue_wait_ms_list),
        "gate_observed": {
            "max_active_jobs": max_active[0],
            "max_queued_jobs": max_queued[0],
        },
    }

    # Top-level shims for backward-compatible stats roll-up
    mean_reduction = round(sum(reduction_pcts) / len(reduction_pcts), 1) if reduction_pcts else 0.0
    mean_optimized_size = int(sum(optimized_sizes) / len(optimized_sizes)) if optimized_sizes else 0
    most_common_method = Counter(methods).most_common(1)[0][0] if methods else ""

    return {
        **base,
        "measurement": measurement_to_dict(m),
        "tool_invocations": all_invocations,
        "reduction_pct": mean_reduction,
        "method": most_common_method,
        "optimized_size": mean_optimized_size,
        "load": load_block,
    }


async def run_load(
    cases: list[Case],
    *,
    n_concurrent: int,
    semaphore_size: int,
    queue_depth: int,
    memory_budget_mb: int = 0,
) -> list[dict[str, Any]]:
    """Sequentially run load cases; N concurrent requests within each case.

    Sequential per-case ensures wall-time isolation between cases — the
    same rationale as quick/timing modes. Parallel within-case is the
    whole point: N requests race through a shared gate.
    """
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = await _run_one_load_case(
                case,
                n_concurrent=n_concurrent,
                semaphore_size=semaphore_size,
                queue_depth=queue_depth,
                memory_budget_mb=memory_budget_mb,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("case %s unexpected failure: %s", case.case_id, exc)
            results.append(
                {
                    "case_id": case.case_id,
                    "name": case.name,
                    "bucket": case.bucket,
                    "format": case.fmt,
                    "preset": case.preset,
                    "iteration": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def run_load_sync(
    cases: list[Case],
    *,
    n_concurrent: int,
    semaphore_size: int,
    queue_depth: int,
    memory_budget_mb: int = 0,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for use from ``bench.runner.cli``."""
    return asyncio.run(
        run_load(
            cases,
            n_concurrent=n_concurrent,
            semaphore_size=semaphore_size,
            queue_depth=queue_depth,
            memory_budget_mb=memory_budget_mb,
        )
    )

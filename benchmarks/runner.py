"""Benchmark runner.

Executes optimization and estimation on each BenchmarkCase,
collecting timing and result data for analysis.
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field

from benchmarks.cases import BenchmarkCase, build_all_cases
from benchmarks.constants import ALL_PRESETS, QualityPreset
from estimation.header_analysis import HeaderInfo, analyze_header
from estimation.heuristics import predict_reduction
from optimizers.router import optimize_image
from schemas import OptimizationConfig
from utils.format_detect import ImageFormat, detect_format


@dataclass
class BenchmarkResult:
    case: BenchmarkCase
    preset_name: str = ""
    # Optimization results
    optimized_size: int = 0
    reduction_pct: float = 0.0
    method: str = ""
    opt_time_ms: float = 0.0
    opt_error: str = ""
    # Throughput
    bytes_per_second: float = 0.0
    # Estimation results
    est_size: int = 0
    est_reduction_pct: float = 0.0
    est_potential: str = ""
    est_confidence: str = ""
    est_time_ms: float = 0.0
    est_error: str = ""
    # Derived
    est_error_pct: float = 0.0  # abs(est_reduction - actual_reduction)


@dataclass
class BenchmarkSuite:
    results: list[BenchmarkResult] = field(default_factory=list)
    total_time_s: float = 0.0
    cases_run: int = 0
    cases_failed: int = 0
    presets_used: list[str] = field(default_factory=list)


def _precompute_headers(cases: list[BenchmarkCase]) -> dict[int, tuple[ImageFormat, HeaderInfo]]:
    """Pre-compute header analysis for each unique case.

    Header analysis (probes, oxipng, quantize) depends only on image data,
    not on quality settings. Computing once and reusing across presets
    avoids redundant work.
    """
    cache: dict[int, tuple[ImageFormat, HeaderInfo]] = {}
    for case in cases:
        key = id(case.data)
        if key not in cache:
            fmt = detect_format(case.data)
            info = analyze_header(case.data, fmt)
            cache[key] = (fmt, info)
    return cache


async def run_single(
    case: BenchmarkCase,
    config: OptimizationConfig,
    preset_name: str = "",
    header_cache: dict[int, tuple[ImageFormat, HeaderInfo]] | None = None,
) -> BenchmarkResult:
    """Run optimize + estimate on a single benchmark case.

    Optimization and estimation run concurrently since they're independent.
    When header_cache is provided, estimation uses pre-computed headers
    instead of re-analyzing the image.
    """
    result = BenchmarkResult(case=case, preset_name=preset_name)

    async def _optimize():
        try:
            t0 = time.perf_counter()
            opt_result = await optimize_image(case.data, config)
            elapsed_s = time.perf_counter() - t0
            result.opt_time_ms = elapsed_s * 1000
            result.optimized_size = opt_result.optimized_size
            result.reduction_pct = opt_result.reduction_percent
            result.method = opt_result.method
            if elapsed_s > 0:
                result.bytes_per_second = len(case.data) / elapsed_s
        except Exception as e:
            result.opt_error = str(e)

    async def _estimate():
        try:
            t0 = time.perf_counter()
            if header_cache and id(case.data) in header_cache:
                fmt, header_info = header_cache[id(case.data)]
                prediction = predict_reduction(header_info, fmt, config)
            else:
                from estimation.estimator import estimate as run_estimate

                est_result = await run_estimate(case.data, config)
                prediction = type(
                    "P",
                    (),
                    {
                        "estimated_size": est_result.estimated_optimized_size,
                        "reduction_percent": est_result.estimated_reduction_percent,
                        "potential": est_result.optimization_potential,
                        "confidence": est_result.confidence,
                    },
                )()
            result.est_time_ms = (time.perf_counter() - t0) * 1000
            result.est_size = prediction.estimated_size
            result.est_reduction_pct = prediction.reduction_percent
            result.est_potential = prediction.potential
            result.est_confidence = prediction.confidence
        except Exception as e:
            result.est_error = str(e)

    await asyncio.gather(_optimize(), _estimate())

    # Estimation accuracy
    if not result.opt_error and not result.est_error:
        result.est_error_pct = abs(result.est_reduction_pct - result.reduction_pct)

    return result


async def run_suite(
    cases: list[BenchmarkCase] | None = None,
    config: OptimizationConfig | None = None,
    presets: list[QualityPreset] | None = None,
    fmt_filter: str | None = None,
    category_filter: str | None = None,
    progress: bool = True,
) -> BenchmarkSuite:
    """Run the full benchmark suite.

    Args:
        cases: Specific cases to run, or None for all.
        config: Optimization config override (used when presets is None).
        presets: Quality presets to run. Defaults to all three.
        fmt_filter: Only run cases matching this format (e.g. "png").
        category_filter: Only run cases matching this category (e.g. "medium").
        progress: Print progress to stderr.
    """
    if cases is None:
        cases = build_all_cases()

    if fmt_filter:
        cases = [c for c in cases if c.fmt == fmt_filter]
    if category_filter:
        cases = [c for c in cases if c.category == category_filter]

    # Determine what to run: explicit presets, single config, or default all presets
    if presets is not None:
        run_list = [(p.name, p.config) for p in presets]
    elif config is not None:
        run_list = [("custom", config)]
    else:
        run_list = [(p.name, p.config) for p in ALL_PRESETS]

    suite = BenchmarkSuite()
    suite.presets_used = [name for name, _ in run_list]

    # Pre-compute header analysis once per unique image (shared across presets)
    header_cache = _precompute_headers(cases)

    # Flatten all (case, preset) combinations for maximum parallelism
    all_tasks_args: list[tuple[BenchmarkCase, OptimizationConfig, str]] = []
    for preset_name, opt_config in run_list:
        for case in cases:
            all_tasks_args.append((case, opt_config, preset_name))

    total = len(all_tasks_args)
    done = 0
    t_start = time.perf_counter()

    # Semaphore limits concurrent tasks; scale with CPU count
    max_workers = min(os.cpu_count() or 4, 12)
    sem = asyncio.Semaphore(max_workers)

    async def _run_with_sem(case, opt_config, preset_name):
        nonlocal done
        async with sem:
            result = await run_single(case, opt_config, preset_name, header_cache)
        done += 1
        if progress:
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"\r  [{done}/{total}] {rate:.0f} cases/s  ETA {eta:.0f}s" + " " * 20,
                end="",
                flush=True,
                file=sys.stderr,
            )
        return result

    results = await asyncio.gather(*[_run_with_sem(c, cfg, pn) for c, cfg, pn in all_tasks_args])

    for result in results:
        suite.results.append(result)
        suite.cases_run += 1
        if result.opt_error:
            suite.cases_failed += 1

    suite.total_time_s = time.perf_counter() - t_start

    if progress:
        print(
            f"\r  Done: {suite.cases_run} cases in {suite.total_time_s:.1f}s" + " " * 40,
            file=sys.stderr,
        )

    return suite

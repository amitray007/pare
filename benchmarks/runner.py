"""Benchmark runner.

Executes optimization and estimation on each BenchmarkCase,
collecting timing and result data for analysis.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field

from benchmarks.cases import BenchmarkCase, build_all_cases
from benchmarks.constants import ALL_PRESETS, QualityPreset
from estimation.estimator import estimate as run_estimate
from optimizers.router import optimize_image
from schemas import OptimizationConfig


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


async def run_single(case: BenchmarkCase, config: OptimizationConfig, preset_name: str = "") -> BenchmarkResult:
    """Run optimize + estimate on a single benchmark case."""
    result = BenchmarkResult(case=case, preset_name=preset_name)

    # Run optimization
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

    # Run estimation
    try:
        t0 = time.perf_counter()
        est_result = await run_estimate(case.data, config)
        result.est_time_ms = (time.perf_counter() - t0) * 1000
        result.est_size = est_result.estimated_optimized_size
        result.est_reduction_pct = est_result.estimated_reduction_percent
        result.est_potential = est_result.optimization_potential
        result.est_confidence = est_result.confidence
    except Exception as e:
        result.est_error = str(e)

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

    total = len(cases) * len(run_list)
    done = 0
    t_start = time.perf_counter()

    for preset_name, opt_config in run_list:
        if progress and len(run_list) > 1:
            print(f"\n  Preset: {preset_name}", file=sys.stderr)

        for i, case in enumerate(cases, 1):
            done += 1
            if progress:
                print(f"\r  [{done}/{total}] {preset_name}: {case.name}...", end="", flush=True, file=sys.stderr)

            result = await run_single(case, opt_config, preset_name=preset_name)
            suite.results.append(result)
            suite.cases_run += 1
            if result.opt_error:
                suite.cases_failed += 1

    suite.total_time_s = time.perf_counter() - t_start

    if progress:
        print(f"\r  Done: {suite.cases_run} cases in {suite.total_time_s:.1f}s" + " " * 40, file=sys.stderr)

    return suite

"""Pare benchmark suite.

Run all benchmarks:
    python -m benchmarks.run

Run with filters:
    python -m benchmarks.run --fmt png
    python -m benchmarks.run --category medium-l
    python -m benchmarks.run --preset high --fmt png
    python -m benchmarks.run --json -o results.json
    python -m benchmarks.run --compare
"""

from benchmarks.cases import BenchmarkCase, build_all_cases
from benchmarks.constants import (
    ALL_PRESETS,
    HIGH,
    LOW,
    MEDIUM,
    PRESETS_BY_NAME,
    QualityPreset,
)
from benchmarks.report import export_json, generate_html_report, print_report
from benchmarks.runner import BenchmarkResult, BenchmarkSuite, run_suite

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkSuite",
    "QualityPreset",
    "HIGH",
    "MEDIUM",
    "LOW",
    "ALL_PRESETS",
    "PRESETS_BY_NAME",
    "build_all_cases",
    "run_suite",
    "print_report",
    "export_json",
    "generate_html_report",
]

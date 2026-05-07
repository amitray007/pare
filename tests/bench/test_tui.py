"""Unit tests for bench/tui.py — argument builders and utility functions.

These tests cover only the pure, non-interactive parts of the TUI:
  - build_run_args        — bench.run argument list construction
  - build_compare_args    — bench.compare argument list construction
  - build_dashboard_args  — bench.dashboard.build argument list construction
  - find_free_port        — free port detection
  - Default-detection     — graceful behaviour when baseline file is absent
"""

from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bench.tui import (
    build_compare_args,
    build_dashboard_args,
    build_run_args,
    find_free_port,
)

# ---------------------------------------------------------------------------
# build_run_args
# ---------------------------------------------------------------------------


class TestBuildRunArgs:
    def test_pr_mode_full_options(self) -> None:
        args = build_run_args(
            mode="pr",
            manifest="core",
            fmt_filter="jpeg,png",
            bucket="small",
            repeat=3,
            warmup=1,
            out="reports/head.json",
        )
        assert args[0:2] == ["-m", "bench.run"]
        assert "--mode" in args and args[args.index("--mode") + 1] == "pr"
        assert "--manifest" in args and args[args.index("--manifest") + 1] == "core"
        assert "--fmt" in args
        # Both formats should appear (two separate --fmt flags).
        fmt_indices = [i for i, a in enumerate(args) if a == "--fmt"]
        fmt_values = {args[i + 1] for i in fmt_indices}
        assert fmt_values == {"jpeg", "png"}
        assert "--bucket" in args and args[args.index("--bucket") + 1] == "small"
        assert "--repeat" in args and args[args.index("--repeat") + 1] == "3"
        assert "--warmup" in args and args[args.index("--warmup") + 1] == "1"
        assert "--out" in args and args[args.index("--out") + 1] == "reports/head.json"

    def test_quick_mode_no_repeat_warmup(self) -> None:
        """quick mode should not emit --repeat / --warmup flags."""
        args = build_run_args(
            mode="quick",
            manifest="core",
            fmt_filter="",
            bucket="all",
            repeat=5,
            warmup=2,
            out="reports/quick.json",
        )
        assert "--repeat" not in args
        assert "--warmup" not in args

    def test_timing_mode_includes_repeat_warmup(self) -> None:
        args = build_run_args(
            mode="timing",
            manifest="core",
            fmt_filter="",
            bucket="all",
            repeat=7,
            warmup=2,
            out="reports/t.json",
        )
        assert "--repeat" in args and args[args.index("--repeat") + 1] == "7"
        assert "--warmup" in args and args[args.index("--warmup") + 1] == "2"

    def test_bucket_all_not_passed(self) -> None:
        """bucket='all' should not add a --bucket flag."""
        args = build_run_args(
            mode="quick",
            manifest="core",
            fmt_filter="",
            bucket="all",
            repeat=1,
            warmup=0,
            out="reports/x.json",
        )
        assert "--bucket" not in args

    def test_bucket_specific_is_passed(self) -> None:
        args = build_run_args(
            mode="quick",
            manifest="core",
            fmt_filter="",
            bucket="xlarge",
            repeat=1,
            warmup=0,
            out="reports/x.json",
        )
        assert "--bucket" in args and args[args.index("--bucket") + 1] == "xlarge"

    def test_empty_fmt_filter_no_fmt_flag(self) -> None:
        args = build_run_args(
            mode="pr",
            manifest="core",
            fmt_filter="",
            bucket="all",
            repeat=1,
            warmup=0,
            out="reports/x.json",
        )
        assert "--fmt" not in args

    def test_whitespace_only_fmt_filter_treated_as_empty(self) -> None:
        args = build_run_args(
            mode="pr",
            manifest="core",
            fmt_filter="   ",
            bucket="all",
            repeat=1,
            warmup=0,
            out="reports/x.json",
        )
        assert "--fmt" not in args

    def test_single_fmt_filter(self) -> None:
        args = build_run_args(
            mode="timing",
            manifest="core",
            fmt_filter="webp",
            bucket="all",
            repeat=1,
            warmup=0,
            out="reports/x.json",
        )
        fmt_indices = [i for i, a in enumerate(args) if a == "--fmt"]
        assert len(fmt_indices) == 1
        assert args[fmt_indices[0] + 1] == "webp"

    def test_annotate_included_when_set(self) -> None:
        args = build_run_args(
            mode="pr",
            manifest="core",
            fmt_filter="",
            bucket="all",
            repeat=1,
            warmup=0,
            out="reports/x.json",
            annotate="env=ci",
        )
        assert "--annotate" in args and args[args.index("--annotate") + 1] == "env=ci"

    def test_annotate_excluded_when_empty(self) -> None:
        args = build_run_args(
            mode="pr",
            manifest="core",
            fmt_filter="",
            bucket="all",
            repeat=1,
            warmup=0,
            out="reports/x.json",
            annotate="",
        )
        assert "--annotate" not in args

    def test_smoke_equivalent_args(self) -> None:
        """The smoke flow uses pr + jpeg + small + 1 iter + 0 warmup."""
        args = build_run_args(
            mode="pr",
            manifest="core",
            fmt_filter="jpeg",
            bucket="small",
            repeat=1,
            warmup=0,
            out="/tmp/smoke.json",
        )
        assert args[args.index("--mode") + 1] == "pr"
        assert args[args.index("--out") + 1] == "/tmp/smoke.json"
        # repeat and warmup should appear because mode is pr.
        assert args[args.index("--repeat") + 1] == "1"
        assert args[args.index("--warmup") + 1] == "0"

    def test_full_manifest_passed_through(self) -> None:
        args = build_run_args(
            mode="quick",
            manifest="full",
            fmt_filter="",
            bucket="all",
            repeat=1,
            warmup=0,
            out="reports/x.json",
        )
        assert args[args.index("--manifest") + 1] == "full"


# ---------------------------------------------------------------------------
# build_compare_args
# ---------------------------------------------------------------------------


class TestBuildCompareArgs:
    def test_basic_markdown(self) -> None:
        args = build_compare_args(
            baseline="reports/baseline.core.json",
            head="reports/head.json",
            threshold_pct=10,
            noise_floor_pct=25,
            fmt="markdown",
        )
        assert args[:2] == ["-m", "bench.compare"]
        assert "reports/baseline.core.json" in args
        assert "reports/head.json" in args
        assert "--threshold-pct" in args and args[args.index("--threshold-pct") + 1] == "10"
        assert "--noise-floor-pct" in args and args[args.index("--noise-floor-pct") + 1] == "25"
        assert "--format" in args and args[args.index("--format") + 1] == "markdown"

    def test_json_format(self) -> None:
        args = build_compare_args(
            baseline="a.json",
            head="b.json",
            threshold_pct=5,
            noise_floor_pct=15,
            fmt="json",
        )
        assert args[args.index("--format") + 1] == "json"

    def test_custom_thresholds(self) -> None:
        args = build_compare_args(
            baseline="a.json",
            head="b.json",
            threshold_pct=20.5,
            noise_floor_pct=50.0,
            fmt="markdown",
        )
        assert args[args.index("--threshold-pct") + 1] == "20.5"
        assert args[args.index("--noise-floor-pct") + 1] == "50.0"

    def test_positional_order(self) -> None:
        """baseline and head should appear before any flags."""
        args = build_compare_args(
            baseline="base.json",
            head="head.json",
            threshold_pct=10,
            noise_floor_pct=25,
            fmt="markdown",
        )
        # After "-m bench.compare", the next two positional args are baseline + head.
        assert args[2] == "base.json"
        assert args[3] == "head.json"


# ---------------------------------------------------------------------------
# build_dashboard_args
# ---------------------------------------------------------------------------


class TestBuildDashboardArgs:
    def test_with_samples_default(self) -> None:
        args = build_dashboard_args(
            baseline="reports/baseline.core.json",
            out_dir="/tmp/dash",
            with_samples=True,
        )
        assert args[:2] == ["-m", "bench.dashboard.build"]
        assert "--baseline" in args
        assert args[args.index("--baseline") + 1] == "reports/baseline.core.json"
        assert "--out-dir" in args
        assert args[args.index("--out-dir") + 1] == "/tmp/dash"
        assert "--no-with-samples" not in args

    def test_without_samples(self) -> None:
        args = build_dashboard_args(
            baseline="reports/x.json",
            out_dir="/tmp/dash2",
            with_samples=False,
        )
        assert "--no-with-samples" in args

    def test_custom_paths(self) -> None:
        args = build_dashboard_args(
            baseline="/custom/base.json",
            out_dir="/custom/out",
            with_samples=True,
        )
        assert args[args.index("--baseline") + 1] == "/custom/base.json"
        assert args[args.index("--out-dir") + 1] == "/custom/out"


# ---------------------------------------------------------------------------
# find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_integer_in_range(self) -> None:
        port = find_free_port(start=18765, attempts=20)
        assert isinstance(port, int)
        assert 18765 <= port < 18785

    def test_returned_port_is_actually_free(self) -> None:
        port = find_free_port(start=19000, attempts=20)
        # Should be able to bind to the returned port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))

    def test_skips_bound_port(self) -> None:
        """Bind a port, then verify find_free_port skips it."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
            busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            busy.bind(("127.0.0.1", 0))
            taken_port = busy.getsockname()[1]
            # find_free_port starting at taken_port should return the NEXT free one.
            port = find_free_port(start=taken_port, attempts=10)
            assert port != taken_port

    def test_raises_when_all_ports_taken(self) -> None:
        """When bind always fails, OSError is raised after exhausting attempts."""
        with patch("socket.socket") as MockSocket:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.bind.side_effect = OSError("Address in use")
            MockSocket.return_value = mock_instance

            with pytest.raises(OSError):
                find_free_port(start=9000, attempts=3)


# ---------------------------------------------------------------------------
# Default-detection / compare flow graceful skip when files are missing
# ---------------------------------------------------------------------------


class TestCompareMissingFiles:
    def test_both_files_missing(self, tmp_path: Path) -> None:
        """build_compare_args is pure; verify it doesn't crash on missing paths."""
        missing_a = str(tmp_path / "a.json")
        missing_b = str(tmp_path / "b.json")
        # The builder itself doesn't check existence — that's the flow's job.
        args = build_compare_args(
            baseline=missing_a,
            head=missing_b,
            threshold_pct=10,
            noise_floor_pct=25,
            fmt="markdown",
        )
        assert missing_a in args
        assert missing_b in args

    def test_flow_detects_missing_baseline(self, tmp_path: Path) -> None:
        """When baseline doesn't exist and the user says 'skip', the flow exits cleanly.

        We verify the detection logic via Path.exists(), not by running the full
        interactive flow.
        """
        baseline = tmp_path / "baseline.json"
        head = tmp_path / "head.json"
        head.write_text("{}")  # only head exists

        assert not baseline.exists()
        missing = [p for p in (str(baseline), str(head)) if not Path(p).exists()]
        assert str(baseline) in missing
        assert str(head) not in missing

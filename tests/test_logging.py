"""Tests for structured JSON logging."""

import json
import logging

from utils.logging import StructuredFormatter, get_logger, setup_logging


def test_structured_log_format():
    """Log output is valid JSON with required fields."""
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="pare.test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="Test error message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["severity"] == "ERROR"
    assert parsed["message"] == "Test error message"
    assert "timestamp" in parsed


def test_log_error_includes_context():
    """Error logs include context dict when provided."""
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="pare.test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="Tool crash",
        args=(),
        exc_info=None,
    )
    record.context = {"tool": "pngquant", "format": "png", "file_size": 1024}
    record.request_id = "test-uuid-123"

    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["context"]["tool"] == "pngquant"
    assert parsed["context"]["format"] == "png"
    assert parsed["request_id"] == "test-uuid-123"


def test_log_severity_mapping():
    """Python log levels map to Cloud Logging severity."""
    formatter = StructuredFormatter()
    levels = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }
    for level, expected in levels.items():
        record = logging.LogRecord(
            name="pare.test",
            level=level,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["severity"] == expected


def test_get_logger_namespace():
    """get_logger returns logger under 'pare' namespace."""
    logger = get_logger("test.module")
    assert logger.name == "pare.test.module"


def test_log_exception_includes_traceback():
    """Exception info included as traceback field."""
    formatter = StructuredFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="pare.test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="Caught exception",
        args=(),
        exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "traceback" in parsed
    assert "ValueError: test error" in parsed["traceback"]

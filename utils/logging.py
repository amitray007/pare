import json
import logging
import sys
from datetime import datetime, timezone

from config import settings


class StructuredFormatter(logging.Formatter):
    """JSON formatter for Google Cloud Logging compatibility.

    Outputs one JSON object per line with fields:
    - severity: Maps Python levels to Cloud Logging severity
    - message: Human-readable message
    - timestamp: ISO 8601 with timezone
    - request_id: From log record extras (if available)
    - context: Additional structured data
    """

    SEVERITY_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": self.SEVERITY_MAP.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        if hasattr(record, "context"):
            log_entry["context"] = record.context

        if record.exc_info and record.exc_info[0]:
            log_entry["traceback"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging() -> logging.Logger:
    """Configure structured JSON logging.

    Call once at application startup (in main.py lifespan).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger("pare")
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Suppress noisy library loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'pare' namespace."""
    return logging.getLogger(f"pare.{name}")

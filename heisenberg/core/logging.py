import logging
import sys
import json
import time
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context var for correlation ID
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "correlation_id": _correlation_id.get(),
        }
        
        # Add extra fields if they exist
        if hasattr(record, "extra"):
             log_record.update(record.extra) # type: ignore

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)

def setup_logging(level: str = "INFO"):
    root = logging.getLogger()
    root.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z"))
    root.addHandler(handler)

def set_correlation_id(cid: str):
    _correlation_id.set(cid)

def get_correlation_id() -> Optional[str]:
    return _correlation_id.get()

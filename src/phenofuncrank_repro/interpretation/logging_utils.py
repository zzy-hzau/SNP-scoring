"""Logging and request counters for interpretation."""
import json,logging,os,threading
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime,timezone
from typing import Any
from phenofuncrank_repro.interpretation.settings import settings
trace_id_var=ContextVar("trace_id",default="")
_lock=threading.Lock()
_counters=defaultdict(float)

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(), "logger": record.name,
            "message": record.getMessage(), "trace_id": trace_id_var.get() or None,
            "event": getattr(record, "event", None), "context": getattr(record, "context", None),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("RICE_PH_AGENT_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger

def increment(name: str, value: float = 1.0, **labels: Any) -> None:
    key = (name, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
    with _lock:
        _counters[key] += value

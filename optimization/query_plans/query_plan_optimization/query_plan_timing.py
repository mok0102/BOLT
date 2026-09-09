"""Process-local timing events; enabled only inside an experiment timing scope."""
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TRACE_ENV = "BOLT_QUERY_TIMING_TRACE"
_PROCESS_ID = uuid.uuid4().hex

def event(phase, seconds, **metadata):
    directory = os.environ.get(TRACE_ENV)
    if not directory:
        return
    row = {"phase": phase, "seconds": float(seconds), "pid": os.getpid(),
           "timestamp": datetime.now(timezone.utc).isoformat(), **metadata}
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"{os.getpid()}-{_PROCESS_ID}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")

@contextmanager
def span(phase, **metadata):
    start = time.perf_counter()
    status = "completed"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        event(phase, time.perf_counter() - start, status=status, **metadata)

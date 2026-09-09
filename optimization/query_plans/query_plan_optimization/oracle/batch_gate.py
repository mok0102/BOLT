"""Optional interprocess DB gate enabled only by the batch trajectory entry point."""
import os
from contextlib import contextmanager
from functools import wraps
import time
from query_plan_timing import event

ENV_NAME = "BOLT_BATCH_DB_LOCK"

@contextmanager
def batch_db_slot():
    path = os.environ.get(ENV_NAME)
    if not path:
        yield
        return
    import fcntl
    # Never unlink this file: all workers must lock the same inode, including
    # subprocesses and overlapping batches. Closing releases the lock on errors.
    with open(path, "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def serialized_batch_query(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # Acquire before connection creation and before the SQL execution timer.
        # Release only after the connection context has committed/rolled back.
        start = time.perf_counter()
        waited = 0.0
        try:
            with batch_db_slot():
                waited = time.perf_counter() - start if os.environ.get(ENV_NAME) else 0.0
                return function(*args, **kwargs)
        finally:
            event("db_queue_wait", waited)
            event("db_call_including_queue", time.perf_counter() - start)
    return wrapped

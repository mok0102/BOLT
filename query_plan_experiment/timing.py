"""Per-invocation JSON reports, safe across batch workers and subprocesses."""
import inspect
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from .config import ExperimentConfig  # also exposes the BO support module path
from query_plan_timing import TRACE_ENV, event


def timed_operation(phase):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            cfg = bound.arguments['cfg']
            workload = bound.arguments.get('workload')
            milestone = bound.arguments.get('milestone')
            identity = uuid.uuid4().hex
            group = Path('tasks') / workload if workload else Path('milestones') / str(milestone)
            target = cfg.run_dir / 'timings' / group / f'{phase}_{identity}.json'
            trace = cfg.run_dir / 'timings' / 'traces' / identity
            target.parent.mkdir(parents=True, exist_ok=True)
            trace.mkdir(parents=True, exist_ok=True)
            previous = os.environ.get(TRACE_ENV)
            os.environ[TRACE_ENV] = str(trace)
            started = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            status = 'completed'
            error = None
            try:
                return function(*args, **kwargs)
            except BaseException as exc:
                status = 'failed'
                error = type(exc).__name__
                raise
            finally:
                elapsed = time.perf_counter() - start
                if previous is None:
                    os.environ.pop(TRACE_ENV, None)
                else:
                    os.environ[TRACE_ENV] = previous
                totals = defaultdict(float)
                counts = defaultdict(int)
                query_results = defaultdict(int)
                files = sorted(trace.glob('*.jsonl'))
                for path in files:
                    for line in path.read_text().splitlines():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # a forcibly terminated child can leave a partial last line
                        totals[row['phase']] += row['seconds']
                        counts[row['phase']] += 1
                        if row['phase'] == 'db_query_execution':
                            query_results[row.get('result', 'unknown')] += 1
                if counts['cache_hit'] and status == 'completed':
                    status = 'reused'
                report = {
                    'schema_version': 1, 'experiment_id': cfg.experiment_id,
                    'operation': phase, 'workload': workload, 'milestone': milestone,
                    'run_id': bound.arguments.get('run_id'),
                    'model_path': str(bound.arguments.get('model_path')) if bound.arguments.get('model_path') else None,
                    'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat(),
                    'wall_seconds': elapsed, 'status': status, 'error_type': error,
                    'phase_seconds': dict(totals), 'phase_counts': dict(counts),
                    'db_result_counts': dict(query_results),
                    'shared_training_tasks': cfg.train_task_workloads()[:milestone] if milestone is not None else [],
                    'trace_files': [str(p) for p in files],
                    'note': 'Phase spans may overlap/nest: do not sum them as total wall time. Training is shared per milestone, not charged to each task.'}
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(report, indent=2))
                temporary.replace(target)
                event('nested_operation', elapsed, operation=phase, report_path=str(target), status=status)
        return wrapped
    return decorate

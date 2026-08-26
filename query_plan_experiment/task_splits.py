"""Name-keyed task splits for the query-plan domain (analogous to
optimization/peptides/apex_oracle/task_splits.py, but workloads are named
strings like "CEB_8A94", not a 0-999 integer index, so a range()-based design
doesn't transfer).

Real discrepancy, resolved here as a named, revisable assumption (paper says
2933 training + 99 held-out = 3032; tasks/ceb_tasks.txt actually has 3033
unique entries, verified precisely -- one extra, no train/heldout annotation
at all): sort every workload name deterministically (by parsed
(number, letter, number) key, NOT on-disk file order, which isn't documented
as meaningful), take the first N_HELDOUT as the held-out set, the next
N_TRAIN as the training workload (disjoint from held-out by construction).
The remaining ~1508 tasks are unused by this stage's closed-loop BO chain --
the paper's larger 2933-query corpus feeds a self-augmentation step distinct
from the milestone-training loop this stage targets, out of scope here. See
imp_plan/02_query_plan_reimplementation_plan.md.
"""

from __future__ import annotations

import re
from pathlib import Path

BOLT_ROOT = Path(__file__).resolve().parents[1]
TASKS_FILE = BOLT_ROOT / "optimization" / "query_plans" / "tasks" / "ceb_tasks.txt"

N_HELDOUT = 99
N_TRAIN = 1426

_TASK_RE = re.compile(r"^CEB_(\d+)([A-Za-z]+)(\d+)$")


def _sort_key(name: str) -> tuple[int, str, int]:
    m = _TASK_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized workload name format: {name!r}")
    number, letter, suffix = m.groups()
    return int(number), letter, int(suffix)


def load_all_workloads() -> list[str]:
    names = [line.strip() for line in TASKS_FILE.read_text().splitlines() if line.strip()]
    assert len(names) == len(set(names)), "ceb_tasks.txt contains duplicate workload names"
    return sorted(names, key=_sort_key)


def heldout_workloads() -> list[str]:
    """The 99-query held-out set (Table 1) -- first 99 of the sorted list."""
    return load_all_workloads()[:N_HELDOUT]


def train_workloads() -> list[str]:
    """The 1426-task training workload (the full closed-loop BO chain) --
    next 1426 after the held-out set, disjoint from it by construction."""
    return load_all_workloads()[N_HELDOUT : N_HELDOUT + N_TRAIN]

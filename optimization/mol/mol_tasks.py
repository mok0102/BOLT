"""Mol's task-identity module: the analogue of peptide's
optimization/peptides/apex_oracle/refseqs.py (the ordered reference list) +
apex_oracle/task_splits.py (heldout ranges over that list) combined, per the
spec's "Task identity" section -- a (protein, seed ligand) task fits the
peptide-style "bare int index into a global ordered list" model only by mol
defining its own ordered manifest and using row position as the index.

Source of truth: optimization/mol/task_manifest/mol_task_manifest.tsv (Milestone
2, 950 rows: 900 train then 50 heldout, in that fixed order -- never reordered
here, since "milestone m" downstream means "tasks 0..m-1" exactly as it does for
peptide's config.py::train_task_range()).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent / "task_manifest" / "mol_task_manifest.tsv"

N_TRAIN_TASKS = 900
N_HELDOUT_TASKS = 50


@dataclass(frozen=True)
class MolTask:
    task_idx: int
    split: str  # "train" or "heldout"
    target_id: str
    target_name: str
    sequence: str
    seed_smiles: str
    seed_selfies: str
    seed_p_median: float
    best_known_p_vocab_ok: float


_TASKS: list[MolTask] | None = None


def load_tasks() -> list[MolTask]:
    global _TASKS
    if _TASKS is not None:
        return _TASKS
    tasks = []
    with open(MANIFEST_PATH, newline="") as f:
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
            tasks.append(MolTask(
                task_idx=idx,
                split=row["split"],
                target_id=row["target_id"],
                target_name=row["target_name"],
                sequence=row["sequence"],
                seed_smiles=row["seed_canonical_smiles"],
                seed_selfies=row["seed_selfies"],
                seed_p_median=float(row["seed_p_median"]),
                best_known_p_vocab_ok=float(row["best_known_p_vocab_ok"]),
            ))
    if len(tasks) != N_TRAIN_TASKS + N_HELDOUT_TASKS:
        raise RuntimeError(
            f"expected {N_TRAIN_TASKS + N_HELDOUT_TASKS} tasks in {MANIFEST_PATH}, found {len(tasks)} "
            "-- manifest must not be silently truncated or regenerated with a different scale "
            "without updating N_TRAIN_TASKS/N_HELDOUT_TASKS here."
        )
    for i, t in enumerate(tasks):
        expected_split = "train" if i < N_TRAIN_TASKS else "heldout"
        if t.split != expected_split:
            raise RuntimeError(
                f"task {i} has split={t.split!r}, expected {expected_split!r} -- "
                "manifest row order is the task index; it must not be reordered or shuffled."
            )
    _TASKS = tasks
    return _TASKS


def get_task(task_idx: int) -> MolTask:
    return load_tasks()[task_idx]


def train_task_range() -> range:
    return range(0, N_TRAIN_TASKS)


def heldout_tasks() -> list[int]:
    return list(range(N_TRAIN_TASKS, N_TRAIN_TASKS + N_HELDOUT_TASKS))

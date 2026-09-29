"""All peptide-domain science for eval2, in one place.

This module is the sole owner of what makes a candidate a *peptide*: how a
raw LLM generation is parsed into a sequence, whether that sequence is
feasible (edit-distance similarity to the task's reference), how it is
scored (the APEX oracle), the on-disk init-pool format, and how a finished
BO trajectory is reduced to a best-so-far number or curve.

The compute engines (compute/generate_raw.py, incumbent.py,
fixed_target_bo.py) reach the domain only through the module-level functions
below -- no REFERENCE_SEQUENCE lookup or similarity formula is allowed to
leak into an engine. That is the seam: adding a second domain later means
writing a module with this same function surface and choosing between them
at the top of each engine, not threading a callback dataclass through every
call site.

Imports peptide_experiment.* freely -- eval2 depends on the production
package, never the reverse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[3]

# make_initialization_data.extract_sequence lives next to the fine-tuning
# sampling scripts rather than in an importable package. This is the only
# sys.path manipulation in eval2; everything else resolves normally under
# `python -m experiments.eval2.cli` from the repo root.
for _p in (BOLT_ROOT, BOLT_ROOT / "fine-tuning" / "peptides" / "sampled_output_from_ft"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

NAME = "peptide"
OBJECTIVE_LABEL = "MIC (lower = more potent)"
TASK_SETS = ("trainset", "heldout", "heldout50", "heldout100")


def load_config(path: str):
    from peptide_experiment.config import load_config as _load_config

    return _load_config(path)


def task_indices(cfg, task_set: str) -> list[int]:
    """trainset: the fixed subset every milestone's checkpoint has already
    been trained on. heldout/heldout50/heldout100: cfg's 20-/50-/100-task
    held-out lists (all three respect heldout_tasks_override)."""
    if task_set == "trainset":
        return list(range(min(cfg.milestones)))
    if task_set == "heldout":
        return list(cfg.heldout_tasks("heldout20"))
    if task_set == "heldout50":
        return list(cfg.heldout_tasks("heldout50"))
    if task_set == "heldout100":
        return list(cfg.heldout_tasks("heldout100"))
    raise ValueError(f"unknown task_set {task_set!r}, expected one of {TASK_SETS}")


def raw_attempt_glob(task_idx: int) -> str:
    return f"task_{task_idx:04d}_sampled_attempt*.jsonl"


def trajectory_csv_name(task_idx: int) -> str:
    return f"task_{task_idx:04d}.csv"


def load_raw_candidates(raw_dir: Path, task_idx: int) -> list[str]:
    """Every sequence the checkpoint generated for this task, in generation
    order across attempts -- unfiltered and undeduplicated, so a caller can
    recover the draw index at which each accepted candidate appeared."""
    from make_initialization_data import extract_sequence

    attempt_files = sorted(
        raw_dir.glob(raw_attempt_glob(task_idx)),
        key=lambda p: int(p.stem.rsplit("attempt", 1)[1]),
    )
    sequences: list[str] = []
    for f in attempt_files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for answer in record.get("generated_answers", []):
                if answer is None:
                    continue
                seq = extract_sequence(str(answer))
                if seq:
                    sequences.append(seq)
    return sequences


def dedup_key(seq: str) -> str:
    return seq


def similarity(seq, reference: str) -> float:
    """Edit-distance similarity to the task's reference sequence -- the same
    formula as peptide_experiment/steps.py::_ensure_constraint_feasible.

    str(seq) matters: a raw LOLBO trajectory CSV's train_x column can pick up
    a non-string dtype from pandas' type inference when the column mixes
    genuine sequences with other row shapes, and edit_distance() rejects a
    bare float.
    """
    from Levenshtein import distance as edit_distance

    length = len(reference)
    return (length - edit_distance(str(seq), reference)) / length


def is_feasible(cfg, task_idx: int, seq: str) -> bool:
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    return similarity(seq, REFERENCE_SEQUENCE[task_idx]) >= cfg.similarity_threshold


def sample_raw_proposals(cfg, checkpoint_dir, task_idx: int, work_dir: Path, sampling_pool=None) -> None:
    from peptide_experiment.steps import sample_and_build_init

    sample_and_build_init(
        cfg, checkpoint_dir, task_idx, work_dir,
        temperature=cfg.eval_raw_temperature, temperature_step=cfg.eval_raw_temperature_step,
        sampling_pool=sampling_pool,
    )


def score_candidates(cfg, task_idx: int, candidates: list[str]) -> list[tuple[str, float]]:
    """(candidate, value) with the maximize convention every trajectory CSV in
    this repo uses: value = -MIC. Peptide's oracle never censors, so there is
    no per-candidate usability flag to carry alongside the value."""
    if not candidates:
        return []
    from apex_oracle import apex_wrapper

    values = list(-apex_wrapper(candidates)[:, 0])
    return list(zip(candidates, values))


def read_existing_pool(work_dir: Path, task_idx: int) -> tuple[int, dict] | None:
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    if not (init_path.exists() and scores_path.exists()):
        return None
    pool_size = sum(1 for line in init_path.read_text().splitlines() if line.strip())
    return pool_size, {"init_path": init_path, "scores_path": scores_path}


def write_init_pool(work_dir: Path, task_idx: int, scored: list[tuple[str, float]]) -> dict:
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    work_dir.mkdir(parents=True, exist_ok=True)
    init_path.write_text("\n".join(seq for seq, _ in scored) + "\n")
    scores_path.write_text("\n".join(f"{value:.8f}" for _, value in scored) + "\n")
    return {"init_path": init_path, "scores_path": scores_path}


def run_bo(cfg, task_idx: int, work_dir: Path, run_id: str, **kwargs) -> Path:
    from peptide_experiment.steps import run_bo as _run_bo

    return _run_bo(cfg, task_idx, work_dir, run_id=run_id, **kwargs)


def read_pool_size(bo_dir: Path, task_idx: int) -> int:
    path = bo_dir / f"task_{task_idx:04d}_init.txt"
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def read_best_feasible_incumbent(bo_dir: Path, task_idx: int) -> float | None:
    """Best (lowest) MIC in the built init pool itself, i.e. the
    generation-time incumbent before any BO acquisition."""
    path = bo_dir / f"task_{task_idx:04d}_scores.csv"
    if not path.exists():
        return None
    scores = [float(x) for x in path.read_text().splitlines() if x.strip()]
    return -max(scores) if scores else None


def best_objective_at_k(cfg, task_idx: int, csv_path: Path, pool_size: int, k: int) -> float | None:
    """Best (lowest) MIC among the first pool_size+k logged rows.

    LOLBO's collected-data CSV logs every candidate it ever evaluates,
    including ones that violate the similarity constraint. A raw max() over
    train_y can therefore pick up a wildly out-of-distribution "candidate"
    that the raw APEX oracle happens to score unrealistically well -- a real,
    confirmed failure mode (an unrelated 22-residue sequence at
    similarity=-0.5 was once reported as an arm's best result for one task).
    So the max is restricted to feasible rows; if none of the first pool_size+k
    rows are feasible, this returns None (treated as missing, like any other
    gap -- never a silent fallback to an infeasible "best").
    """
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    df = pd.read_csv(csv_path)
    row_idx = min(pool_size + k, len(df))
    if row_idx <= 0 or df.empty:
        return None
    window = df.iloc[:row_idx]
    reference = REFERENCE_SEQUENCE[task_idx]
    window = window[window["train_x"].apply(lambda seq: similarity(seq, reference) >= cfg.similarity_threshold)]
    if window.empty:
        return None
    return -window["train_y"].max()  # train_y = -MIC (maximized); MIC = -train_y, lower is better


def running_best_series(cfg, task_idx: int, csv_path: Path, init_size: int) -> pd.Series:
    """The dense, incremental version of best_objective_at_k -- a running max
    of train_y (infeasible rows masked to -inf first, so they can never win),
    indexed from oracle_calls=0 (the init pool itself, row init_size-1).

    Deliberately diverges from best_objective_at_k past the trajectory's
    actual length: that function clamps (repeats the final value forever,
    since row_idx = min(pool_size+k, len(df))), which is right for a single
    point lookup. A dense per-task curve instead simply has no entry past
    len(df)-init_size -- a task whose run terminated early should visibly end
    there, not be silently extended flat to look like it ran the full budget.
    Verified to match best_objective_at_k at every in-range k.
    """
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    df = pd.read_csv(csv_path)
    reference = REFERENCE_SEQUENCE[task_idx]
    feasible = df["train_x"].apply(lambda seq: similarity(seq, reference) >= cfg.similarity_threshold)

    y = df["train_y"].where(feasible, other=float("-inf"))
    running_max = y.cummax()
    if len(running_max) < init_size:
        return pd.Series(dtype=float)
    tail = running_max.iloc[init_size - 1 :].reset_index(drop=True)
    # inf (an all-infeasible window) becomes NaN, matching best_objective_at_k's
    # "report missing rather than fall back to an infeasible best" convention.
    series = (-tail).replace([float("inf")], float("nan"))
    series.index.name = "oracle_calls"
    return series

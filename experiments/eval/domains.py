"""Domain abstraction so experiments/eval/'s compute engines
(generate_raw_proposals.py, incumbent_vs_pool_size.py,
fixed_target_rejection_bo.py) and the paper_labels.py-driven fig_*.py/
tab_*.py scripts can run against either the peptide domain
(peptide_experiment/apex_oracle) or the query-plan domain
(query_plan_experiment/DatabaseObjective) without duplicating the
dedup/feasibility/scoring/reporting logic per domain.

Both domain packages were explicitly written to "mirror peptide_experiment's
shape" (query_plan_experiment's own docstrings), so config field names
(milestones, oracle_budget, init_size, table_k_checkpoints, run_dir,
checkpoints_dir, milestone_checkpoint_dir) already match 1:1 -- what
genuinely differs, and what this module abstracts over, is: task identity
(int index vs. workload string), whether a pre-oracle feasibility constraint
exists (peptide: similarity>=threshold; query-plan: none), the init-pool
file format (peptide: two files, init.txt+scores.csv; query-plan: one
{workload}_init.csv with an x,y,censoring schema), and whether a candidate's
usability is knowable before scoring (peptide: yes; query-plan: no --
DatabaseObjective can only report censoring, i.e. a timed-out query, after
actually running it).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable

import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))
sys.path.insert(0, str(BOLT_ROOT / "fine-tuning" / "peptides" / "sampled_output_from_ft"))


@dataclass
class ScoredCandidate:
    candidate: Any
    value: float  # maximize convention (train_y = -MIC | -runtime), matching every trajectory CSV in this repo
    usable: bool = True  # False iff the domain's oracle censored this candidate (query-plan only; always True for peptide)


@dataclass
class BuiltPool:
    pool_size: int
    draws_used: int
    run_bo_kwargs: dict  # forwarded as **kwargs into domain.run_bo()


@dataclass(frozen=True)
class Domain:
    name: str
    task_sets: tuple[str, ...]
    objective_label: str
    load_config: Callable[[str], Any]
    task_indices: Callable[[Any, str], list]
    raw_attempt_glob: Callable[[Any], str]
    trajectory_csv_name: Callable[[Any], str]  # filename of run_bo()'s per-task trajectory CSV, within its work_dir
    load_raw_candidates: Callable[[Path, Any], list]
    dedup_key: Callable[[Any], Hashable]
    is_feasible: Callable[[Any, Any, Any], bool]
    sample_raw_proposals: Callable[[Any, Path, Any, Path], None]
    score_candidates: Callable[[Any, Any, list], list[ScoredCandidate]]
    read_existing_pool: Callable[[Path, Any], tuple[int, dict] | None]
    write_init_pool: Callable[[Path, Any, list], dict]
    run_bo: Callable[..., Path]
    best_objective_at_k: Callable[[Any, Any, Path, int, int], float | None]
    running_best_series: Callable[[Any, Any, Path, int], pd.Series]


def _running_best_series_from_mask(df: pd.DataFrame, usable_mask: pd.Series, init_size: int) -> pd.Series:
    """Shared by both domains: the dense, incremental version of
    aggregate.py's best_mic_at_k/best_runtime_at_k -- a running max of
    train_y (masking unusable/infeasible rows to -inf first, so they can
    never win), starting at oracle_calls=0 (the init pool itself, row
    init_size-1) rather than one value per fixed k checkpoint. inf (an
    all-unusable window) is reported as NaN, matching best_*_at_k's own
    "return None rather than fall back to an infeasible/censored best"
    convention.

    Deliberately diverges from best_*_at_k for k beyond the trajectory's
    actual length: best_*_at_k clamps (repeats the final value forever,
    since its row_idx = min(init_size+k, len(df))), which is the right call
    for a single point lookup. This function instead simply has no entry
    past len(df)-init_size -- for a dense per-task curve (fig_main_bo.py), a
    task whose run terminated early should visibly end there, not be
    silently extended flat to look like it ran the full nominal budget.
    Verified to match best_*_at_k exactly at every in-range k (real-data and
    synthetic regression checks, see plan verification notes)."""
    y = df["train_y"].where(usable_mask, other=float("-inf"))
    running_max = y.cummax()
    if len(running_max) < init_size:
        return pd.Series(dtype=float)
    tail = running_max.iloc[init_size - 1 :].reset_index(drop=True)
    series = (-tail).replace([float("inf")], float("nan"))
    series.index.name = "oracle_calls"
    return series


# ---------------------------------------------------------------- peptide --

def _peptide_task_indices(cfg, task_set: str) -> list[int]:
    """trainset: the fixed subset every milestone's checkpoint has already
    been trained on (range(min(milestones))). heldout/heldout100: cfg's
    20-/100-task held-out lists (both respect heldout_tasks_override)."""
    if task_set == "trainset":
        return list(range(min(cfg.milestones)))
    if task_set == "heldout":
        return list(cfg.heldout_tasks("heldout20"))
    if task_set == "heldout100":
        return list(cfg.heldout_tasks("heldout100"))
    raise ValueError(f"unknown task_set {task_set!r} for domain=peptide, expected one of ('trainset','heldout','heldout100')")


def _peptide_load_raw_candidates(raw_dir: Path, task_idx: int) -> list[str]:
    import json

    from make_initialization_data import extract_sequence

    attempt_files = sorted(
        raw_dir.glob(f"task_{task_idx:04d}_sampled_attempt*.jsonl"),
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


def _peptide_similarity(seq, reference: str) -> float:
    """Same edit-distance-based formula as
    peptide_experiment/steps.py::_ensure_constraint_feasible and
    peptide_experiment/aggregate.py::_similarity. str(seq) matters here (not
    just in the trainer's own written init.txt lines, which are always
    already str): a raw LOLBO trajectory CSV's train_x column can pick up a
    non-string dtype from pandas' own type inference when the column mixes
    genuine sequences with other row shapes, and edit_distance() rejects a
    bare float."""
    from Levenshtein import distance as edit_distance

    length = len(reference)
    return (length - edit_distance(str(seq), reference)) / length


def _peptide_is_feasible(cfg, task_idx: int, seq: str) -> bool:
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    return _peptide_similarity(seq, REFERENCE_SEQUENCE[task_idx]) >= cfg.similarity_threshold


def _peptide_sample_raw_proposals(cfg, checkpoint_dir, task_idx: int, work_dir: Path) -> None:
    from peptide_experiment.steps import sample_and_build_init

    sample_and_build_init(cfg, checkpoint_dir, task_idx, work_dir)


def _peptide_score_candidates(cfg, task_idx: int, candidates: list[str]) -> list[ScoredCandidate]:
    if not candidates:
        return []
    from apex_oracle import apex_wrapper

    values = list(-apex_wrapper(candidates)[:, 0])
    return [ScoredCandidate(c, v, usable=True) for c, v in zip(candidates, values)]


def _peptide_read_existing_pool(work_dir: Path, task_idx: int) -> tuple[int, dict] | None:
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    if not (init_path.exists() and scores_path.exists()):
        return None
    pool_size = sum(1 for line in init_path.read_text().splitlines() if line.strip())
    return pool_size, {"init_path": init_path, "scores_path": scores_path}


def _peptide_write_init_pool(work_dir: Path, task_idx: int, scored: list[ScoredCandidate]) -> dict:
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    work_dir.mkdir(parents=True, exist_ok=True)
    init_path.write_text("\n".join(c.candidate for c in scored) + "\n")
    scores_path.write_text("\n".join(f"{c.value:.8f}" for c in scored) + "\n")
    return {"init_path": init_path, "scores_path": scores_path}


def _peptide_run_bo(cfg, task_idx: int, work_dir: Path, run_id: str, **kwargs) -> Path:
    from peptide_experiment.steps import run_bo

    return run_bo(cfg, task_idx, work_dir, run_id=run_id, **kwargs)


def _peptide_best_objective_at_k(cfg, task_idx: int, csv_path: Path, pool_size: int, k: int) -> float | None:
    from apex_oracle.refseqs import REFERENCE_SEQUENCE
    from peptide_experiment.aggregate import best_mic_at_k

    return best_mic_at_k(
        csv_path, pool_size, k,
        reference_sequence=REFERENCE_SEQUENCE[task_idx], similarity_threshold=cfg.similarity_threshold,
    )


def _peptide_running_best_series(cfg, task_idx: int, csv_path: Path, init_size: int) -> pd.Series:
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    df = pd.read_csv(csv_path)
    reference = REFERENCE_SEQUENCE[task_idx]
    usable = df["train_x"].apply(lambda seq: _peptide_similarity(seq, reference) >= cfg.similarity_threshold)
    return _running_best_series_from_mask(df, usable, init_size)


def _build_peptide_domain() -> Domain:
    from peptide_experiment.config import load_config

    return Domain(
        name="peptide",
        task_sets=("trainset", "heldout", "heldout100"),
        objective_label="MIC (lower = more potent)",
        load_config=load_config,
        task_indices=_peptide_task_indices,
        raw_attempt_glob=lambda task_idx: f"task_{task_idx:04d}_sampled_attempt*.jsonl",
        trajectory_csv_name=lambda task_idx: f"task_{task_idx:04d}.csv",
        load_raw_candidates=_peptide_load_raw_candidates,
        dedup_key=lambda seq: seq,
        is_feasible=_peptide_is_feasible,
        sample_raw_proposals=_peptide_sample_raw_proposals,
        score_candidates=_peptide_score_candidates,
        read_existing_pool=_peptide_read_existing_pool,
        write_init_pool=_peptide_write_init_pool,
        run_bo=_peptide_run_bo,
        best_objective_at_k=_peptide_best_objective_at_k,
        running_best_series=_peptide_running_best_series,
    )


# -------------------------------------------------------------- query_plan --

def _qp_task_indices(cfg, task_set: str) -> list[str]:
    """trainset: the fixed subset every milestone's checkpoint has already
    been trained on (mirrors peptide's range(min(milestones)) convention).
    heldout: cfg's own heldout_tasks list (respects heldout_tasks_override)."""
    if task_set == "trainset":
        from query_plan_experiment import task_splits

        return task_splits.train_workloads()[: min(cfg.milestones)]
    if task_set == "heldout":
        return list(cfg.heldout_tasks)
    raise ValueError(f"unknown task_set {task_set!r} for domain=query_plan, expected one of ('trainset','heldout')")


def _qp_load_raw_candidates(raw_dir: Path, workload: str) -> list[list[int]]:
    import json

    attempt_files = sorted(
        raw_dir.glob(f"{workload}_sampled_attempt*.jsonl"),
        key=lambda p: int(p.stem.rsplit("attempt", 1)[1]),
    )
    candidates: list[list[int]] = []
    for f in attempt_files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("task") != workload:
                continue
            for candidate in record.get("generated_answers", []):
                if not isinstance(candidate, list) or not candidate:
                    continue  # unparseable LLM output, mirrors steps.py::sample_and_build_init's own filter
                candidates.append(list(candidate))
    return candidates


def _qp_sample_raw_proposals(cfg, checkpoint_dir, workload: str, work_dir: Path) -> None:
    from query_plan_experiment.steps import sample_and_build_init

    sample_and_build_init(cfg, checkpoint_dir, workload, work_dir)


def _qp_score_candidates(cfg, workload: str, candidates: list[list[int]]) -> list[ScoredCandidate]:
    if not candidates:
        return []
    from your_tasks.your_objective_functions import DatabaseObjective  # noqa: E402 -- relies on config.py's sys.path insert

    oracle = DatabaseObjective(
        workload_name=workload,
        worst_runtime_observed=2 * cfg.query_timeout_secs,
        timeout=cfg.query_timeout_secs,
        which_language=cfg.which_query_language,
    )
    ys, censoring = oracle.query_black_box([list(c) for c in candidates])
    return [ScoredCandidate(c, y, usable=(cn == 0)) for c, y, cn in zip(candidates, ys, censoring)]


def _qp_read_existing_pool(work_dir: Path, workload: str) -> tuple[int, dict] | None:
    init_csv_path = work_dir / f"{workload}_init.csv"
    if not init_csv_path.exists():
        return None
    pool_size = sum(1 for _ in init_csv_path.read_text().splitlines()) - 1  # minus header row
    return pool_size, {"init_csv_path": init_csv_path}


def _qp_write_init_pool(work_dir: Path, workload: str, scored: list[ScoredCandidate]) -> dict:
    from query_plan_experiment.steps import _write_init_csv

    init_csv_path = work_dir / f"{workload}_init.csv"
    xs = [list(c.candidate) for c in scored]
    ys = [c.value for c in scored]
    censoring = [0 if c.usable else 1 for c in scored]
    _write_init_csv(init_csv_path, xs, ys, censoring)
    return {"init_csv_path": init_csv_path}


def _qp_run_bo(cfg, workload: str, work_dir: Path, run_id: str, **kwargs) -> Path:
    from query_plan_experiment.steps import run_bo

    return run_bo(cfg, workload, work_dir, run_id=run_id, **kwargs)


def _qp_best_objective_at_k(cfg, workload: str, csv_path: Path, pool_size: int, k: int) -> float | None:
    from query_plan_experiment.aggregate import best_runtime_at_k

    return best_runtime_at_k(csv_path, pool_size, k)


def _qp_running_best_series(cfg, workload: str, csv_path: Path, init_size: int) -> pd.Series:
    df = pd.read_csv(csv_path)
    usable = df["censoring"].astype(float) == 0.0
    return _running_best_series_from_mask(df, usable, init_size)


def _build_query_plan_domain() -> Domain:
    from query_plan_experiment.config import load_config

    return Domain(
        name="query_plan",
        task_sets=("trainset", "heldout"),
        objective_label="query runtime, sec (lower = better)",
        load_config=load_config,
        task_indices=_qp_task_indices,
        raw_attempt_glob=lambda workload: f"{workload}_sampled_attempt*.jsonl",
        trajectory_csv_name=lambda workload: f"{workload}.csv",
        load_raw_candidates=_qp_load_raw_candidates,
        dedup_key=lambda cand: tuple(cand),
        is_feasible=lambda cfg, workload, cand: True,  # no pre-oracle feasibility constraint in this domain
        sample_raw_proposals=_qp_sample_raw_proposals,
        score_candidates=_qp_score_candidates,
        read_existing_pool=_qp_read_existing_pool,
        write_init_pool=_qp_write_init_pool,
        run_bo=_qp_run_bo,
        best_objective_at_k=_qp_best_objective_at_k,
        running_best_series=_qp_running_best_series,
    )


DOMAINS: dict[str, Domain] = {"peptide": _build_peptide_domain(), "query_plan": _build_query_plan_domain()}

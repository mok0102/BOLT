"""Builds Table-11-style and Figure-1/2-style CSVs from held-out eval
outputs. See imp_plan/03_experiment_results_guide.md for how to read these.

Arms are derived from cfg.milestones (+ STBO) rather than hardcoded, so this
also works unmodified against the smoke config's tiny milestones.

Table 11 and Figure 1/2 report two genuinely different quantities (confirmed
via Appendix D.6, p.24): the no-BO milestone eval (paper's Table 11) is an
"initialization only" metric -- k indexes into the init pool itself
(k_max == init_size), zero BO acquisition -- while the BO scaling curve
(paper's Figure 1/2) is a full-BO scaling curve (k = oracle calls made during
acquisition, after an identically-sized init set). build_no_bo_milestone_eval()
reads run_init_only_eval()'s scores files directly; build_bo_scaling_curve()
reads run_heldout_eval()'s full BO trajectory CSVs.

Figure 1/2 assumption (not verified against paper source/plots, only the
caption text): train_x/train_y in the collected-data CSV are appended in
chronological acquisition order, init points first, so "best value within k
oracle calls" is the running max of train_y over rows [0 : init_size + k].
"""

from __future__ import annotations

import pandas as pd
from Levenshtein import distance as edit_distance

from .config import ExperimentConfig
from apex_oracle.refseqs import REFERENCE_SEQUENCE  # noqa: E402 -- relies on config.py's sys.path insert


def _arms(cfg: ExperimentConfig) -> list[str]:
    arms = [f"BOLT-{m}" for m in cfg.milestones]
    if cfg.build_orpt:
        arms += [f"ORPT-{m}" for m in cfg.milestones]
    if cfg.build_mtbo:
        arms += [f"MTBO-{m}" for m in cfg.milestones]
    arms.append("STBO")
    if cfg.build_optformer:
        arms += [f"OptFormer-{m}" for m in cfg.milestones]
    return arms


def _best_score_at_k_from_scores(scores_path, k: int) -> float | None:
    if not scores_path.exists():
        return None
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]
    if not scores:
        return None
    row_idx = min(k, len(scores))
    best_y = max(scores[:row_idx])
    return -best_y  # scores = -MIC (maximized); MIC = -score, lower is better


def _sum_score_across_tasks_init_only(
    cfg: ExperimentConfig, task_set: str, arm: str, k: int
) -> tuple[float, int]:
    arm_dir = cfg.heldout_dir(task_set) / "init_only" / arm
    total = 0.0
    n_missing = 0
    for task_idx in cfg.heldout_tasks(task_set):
        scores_path = arm_dir / f"task_{task_idx:04d}_scores.csv"
        mic = _best_score_at_k_from_scores(scores_path, k)
        if mic is not None:
            total += mic
        else:
            n_missing += 1
    return total, n_missing


def _similarity(seq: str, reference: str) -> float:
    """Same edit-distance-based formula as steps.py::_ensure_constraint_feasible
    and experiments/eval/common.py::similarity()."""
    length = len(reference)
    return (length - edit_distance(str(seq), reference)) / length


def best_mic_at_k(
    csv_path,
    init_size: int,
    k: int,
    reference_sequence: str | None = None,
    similarity_threshold: float | None = None,
) -> float | None:
    """Best (lowest) MIC among the first init_size+k logged rows.

    reference_sequence/similarity_threshold (pass both together, or neither):
    LOLBO's collected-data CSV logs every candidate it ever evaluates,
    including ones that violate the similarity constraint. A raw max() over
    train_y can therefore pick up a wildly out-of-distribution "candidate"
    that the raw APEX oracle happens to score unrealistically well -- a real,
    confirmed failure mode (an unrelated 22-residue sequence at
    similarity=-0.5 was reported as an arm's best result for one task before
    this fix). When given, restrict the max to feasible rows only; if none of
    the first init_size+k rows are feasible, return None (treated as missing,
    like any other gap -- never silently fall back to an infeasible "best").
    """
    df = pd.read_csv(csv_path)
    row_idx = min(init_size + k, len(df))
    if row_idx <= 0 or df.empty:
        return None
    window = df.iloc[:row_idx]
    if reference_sequence is not None and similarity_threshold is not None:
        feasible = window["train_x"].apply(lambda seq: _similarity(seq, reference_sequence) >= similarity_threshold)
        window = window[feasible]
        if window.empty:
            return None
    best_y = window["train_y"].max()
    return -best_y  # train_y = -MIC (maximized); MIC = -train_y, lower is better


def _sum_mic_across_tasks(
    cfg: ExperimentConfig, task_set: str, arm: str, k: int
) -> tuple[float, int]:
    arm_dir = cfg.heldout_dir(task_set) / arm
    total = 0.0
    n_missing = 0
    for task_idx in cfg.heldout_tasks(task_set):
        csv_path = arm_dir / f"task_{task_idx:04d}.csv"
        if not csv_path.exists():
            n_missing += 1
            continue
        mic = best_mic_at_k(
            csv_path,
            cfg.init_size,
            k,
            reference_sequence=REFERENCE_SEQUENCE[task_idx],
            similarity_threshold=cfg.similarity_threshold,
        )
        if mic is not None:
            total += mic
        else:
            n_missing += 1
    return total, n_missing


def build_no_bo_milestone_eval(cfg: ExperimentConfig) -> pd.DataFrame:
    """No-BO milestone eval (paper's Table 11): rows = k (index into the init
    pool itself, zero BO acquisition -- see module docstring), columns = arms,
    cells = summed unnormalized MIC across the 20-task held-out set."""
    arms = _arms(cfg)
    n_tasks = len(cfg.heldout_tasks("heldout20"))
    rows = {}
    for k in cfg.table_k_checkpoints:
        row = {}
        for arm in arms:
            total, n_missing = _sum_score_across_tasks_init_only(cfg, "heldout20", arm, k)
            if n_missing:
                print(f"[no_bo_milestone_eval] arm={arm} k={k}: missing {n_missing}/{n_tasks} task score files")
            row[arm] = total
        rows[k] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "k"

    cfg.aggregate_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.aggregate_dir / "no_bo_milestone_eval.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}")
    return df


def build_bo_scaling_curve(cfg: ExperimentConfig) -> pd.DataFrame:
    """BO scaling curve (paper's Figure 1/2): rows = oracle-call checkpoints,
    columns = arms, cells = summed objective across the 100-task held-out
    set."""
    arms = _arms(cfg)
    n_tasks = len(cfg.heldout_tasks("heldout100"))
    checkpoints = sorted(set(cfg.table_k_checkpoints) | {cfg.oracle_budget})
    rows = {}
    for k in checkpoints:
        row = {}
        for arm in arms:
            total, n_missing = _sum_mic_across_tasks(cfg, "heldout100", arm, k)
            if n_missing:
                print(f"[bo_scaling_curve] arm={arm} k={k}: missing {n_missing}/{n_tasks} task CSVs")
            row[arm] = total
        rows[k] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "oracle_calls"

    cfg.aggregate_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.aggregate_dir / "bo_scaling_curve.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}")
    return df

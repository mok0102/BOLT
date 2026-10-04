"""Heldout evaluation for the mol domain: for one arm checkpoint at one
milestone, run each heldout task's real BO (LLM-sampled init pool, then
mol_run_bo) and report best-objective-at-k.

Mirrors the shape of experiments/eval2/compute/fixed_target_bo.py (read-only
reference, never imported -- isolation contract; experiments/ has no mol
domain) restricted to the two LLM arms mol has: BOLT (SFT checkpoint) and
ORPT-H1 (DPO checkpoint). Baseline arms (STBO/MTBO/POGPE/SGPE/OptFormer/
LLAMBO) are not ported for mol; ARM_CHECKPOINT_DIR below is the one dispatch
point a baseline branch would extend, and an unsupported arm raises rather
than silently falling back to something else.

Layout: one directory per (arm, milestone) under cfg.heldout_dir():
    <run_dir>/heldout/<arm>-<milestone>/
        task_XXXX_init.txt, task_XXXX_scores.csv   (sampled init pool + oracle scores)
        task_XXXX.csv                              (full trajectory: init rows, then BO rows)
        summary.json                               (written by summarize_heldout_eval)
Every step is idempotent per task (an existing task_XXXX.csv is skipped), so
a crashed or interrupted run resumes by re-running the same command.

Parallelism is manual by design (no orchestrator): run N processes with
different CUDA_VISIBLE_DEVICES and --shard k/N, each takes every N-th heldout
task.

Metric (same rule as peptide's best_objective_at_k): best (max) train_y over
the first pool_size + k rows, restricted to rows with Tanimoto(x, seed) >=
tau_mol; None (reported, never silently replaced) when no such row exists.
train_y is the oracle score and is maximized in mol (mol_run_bo uses
minimize=False), so higher is better -- no sign flip.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

from .config import MolExperimentConfig

SUPPORTED_ARMS = ("BOLT", "ORPT-H1")


def arm_checkpoint_dir(cfg: MolExperimentConfig, arm: str, milestone: int) -> Path:
    if arm == "BOLT":
        return cfg.milestone_checkpoint_dir(milestone)
    if arm == "ORPT-H1":
        return cfg.orpt_checkpoint_dir(milestone)
    raise ValueError(
        f"unsupported arm {arm!r} for mol heldout eval; supported: {SUPPORTED_ARMS}. "
        "STBO/MTBO/POGPE/SGPE/OptFormer/LLAMBO have no mol port yet."
    )


def arm_eval_dir(cfg: MolExperimentConfig, arm: str, milestone: int) -> Path:
    return cfg.heldout_dir() / f"{arm}-{milestone}"


def _eval_cfg(cfg: MolExperimentConfig) -> MolExperimentConfig:
    if cfg.eval_init_size is None or cfg.eval_oracle_budget is None:
        raise ValueError(
            "eval_init_size and eval_oracle_budget must both be set in the config for heldout "
            f"eval (got eval_init_size={cfg.eval_init_size}, eval_oracle_budget={cfg.eval_oracle_budget}); "
            "they are deliberately not defaulted to the training init_size/oracle_budget."
        )
    return dataclasses.replace(cfg, init_size=cfg.eval_init_size, oracle_budget=cfg.eval_oracle_budget)


def parse_shard(shard: str) -> tuple[int, int]:
    try:
        k_str, n_str = shard.split("/")
        k, n = int(k_str), int(n_str)
    except ValueError:
        raise ValueError(f"shard must look like 'k/n' (e.g. '0/4'), got {shard!r}") from None
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"shard {shard!r} needs n >= 1 and 0 <= k < n")
    return k, n


def run_heldout_eval(cfg: MolExperimentConfig, arm: str, milestone: int, shard: str = "0/1") -> None:
    from mol_sampling import load_model_and_tokenizer

    from .steps import mol_run_bo, mol_sample_and_build_init

    ckpt = arm_checkpoint_dir(cfg, arm, milestone)
    if not (ckpt / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"{arm}-{milestone} checkpoint not complete at {ckpt} (no model.safetensors.index.json)")
    eval_cfg = _eval_cfg(cfg)
    out_dir = arm_eval_dir(cfg, arm, milestone)
    out_dir.mkdir(parents=True, exist_ok=True)

    k, n = parse_shard(shard)
    tasks = [t for i, t in enumerate(cfg.heldout_tasks()) if i % n == k]
    run_id = f"eval-{arm}-{milestone}"
    print(f"[{run_id}] shard {shard}: {len(tasks)} heldout tasks, init_size={eval_cfg.init_size}, "
          f"oracle_budget={eval_cfg.oracle_budget}, out_dir={out_dir}")

    loaded = None  # (model, tokenizer, device); loaded lazily, once, only if some task needs computing
    try:
        for task_idx in tasks:
            if (out_dir / f"task_{task_idx:04d}.csv").exists():
                print(f"[{run_id} task {task_idx}] trajectory already exists, skipping")
                continue
            if loaded is None:
                loaded = load_model_and_tokenizer(ckpt)
            init_path, scores_path = mol_sample_and_build_init(
                eval_cfg, ckpt, task_idx, out_dir, preloaded_model=loaded,
            )
            mol_run_bo(eval_cfg, task_idx, out_dir, run_id=run_id, init_path=init_path, scores_path=scores_path)
    finally:
        del loaded


def eval_pool_size(init_path: Path) -> int:
    """Number of init rows at the head of the task's trajectory CSV: the init
    file's lines minus any dropped by mol_run_bo's own final vocab-safety gate
    (same filter_vocab_safe, so this matches the CSV's actual layout)."""
    from mol_init_candidates import filter_vocab_safe

    smiles = [line for line in init_path.read_text().splitlines() if line.strip()]
    return sum(filter_vocab_safe(smiles))


def feasible_running_best(task_idx: int, csv_path: Path, tau_mol: float) -> list[float]:
    """Running max of train_y over the CSV's rows, with Tanimoto-infeasible rows
    contributing -inf (so they can never win). Entry i = best over rows [0, i]."""
    from mol_fingerprint import tanimoto_similarity
    from mol_tasks import get_task

    seed_smiles = get_task(task_idx).seed_smiles
    best = float("-inf")
    series: list[float] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            sim = tanimoto_similarity(row["train_x"], seed_smiles)
            if sim is not None and sim >= tau_mol:
                best = max(best, float(row["train_y"]))
            series.append(best)
    return series


def _at_k(series: list[float], pool_size: int, k: int) -> float | None:
    row_idx = min(pool_size + k, len(series))
    if row_idx <= 0:
        return None
    value = series[row_idx - 1]
    return None if value == float("-inf") else value


def best_objective_at_k(cfg: MolExperimentConfig, task_idx: int, csv_path: Path, pool_size: int, k: int) -> float | None:
    return _at_k(feasible_running_best(task_idx, csv_path, cfg.tau_mol), pool_size, k)


def summarize_heldout_eval(cfg: MolExperimentConfig, arm: str, milestone: int) -> dict:
    out_dir = arm_eval_dir(cfg, arm, milestone)
    ks = list(cfg.table_k_checkpoints)
    per_task: dict[int, dict[int, float | None]] = {}
    missing: list[int] = []
    for task_idx in cfg.heldout_tasks():
        csv_path = out_dir / f"task_{task_idx:04d}.csv"
        init_path = out_dir / f"task_{task_idx:04d}_init.txt"
        if not (csv_path.exists() and init_path.exists()):
            missing.append(task_idx)
            continue
        series = feasible_running_best(task_idx, csv_path, cfg.tau_mol)
        pool = eval_pool_size(init_path)
        per_task[task_idx] = {k: _at_k(series, pool, k) for k in ks}

    by_k = {}
    for k in ks:
        vals = [v[k] for v in per_task.values() if v[k] is not None]
        by_k[str(k)] = {
            "mean_best": sum(vals) / len(vals) if vals else None,
            "n_tasks_with_feasible_best": len(vals),
            "n_tasks_without_feasible_row": len(per_task) - len(vals),
        }
    summary = {
        "arm": arm,
        "milestone": milestone,
        "n_heldout_tasks": len(cfg.heldout_tasks()),
        "n_tasks_evaluated": len(per_task),
        "missing_tasks": missing,
        "tau_mol": cfg.tau_mol,
        "by_k": by_k,
        "per_task": {str(t): {str(k): v for k, v in d.items()} for t, d in per_task.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{arm}-{milestone}] {len(per_task)}/{summary['n_heldout_tasks']} tasks evaluated, "
          f"{len(missing)} missing")
    for k in ks:
        d = by_k[str(k)]
        print(f"  k={k}: mean best = {d['mean_best']} "
              f"(n={d['n_tasks_with_feasible_best']}, no-feasible-row={d['n_tasks_without_feasible_row']})")
    return summary

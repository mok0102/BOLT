"""Shared low-level steps reused by both the trajectory chain and held-out
eval: sample+score init candidates for one query-plan workload, and run one
BO trial. Mirrors peptide_experiment/steps.py's shape; see
imp_plan/02_query_plan_reimplementation_plan.md for the full design.
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig

LOLBO_SCRIPTS_DIR = "optimization/query_plans/query_plan_optimization/lolbo_scripts"
WORKLOAD_DIR = "optimization/query_plans/query_plan_optimization/workload"
FINE_TUNING_DIR = "fine-tuning/query_plans"

# sampling_transformer.py generates num_samples completions for one workload
# in a batched loop (sample_batch_size at a time), so unlike peptide's single
# giant model.generate() call this doesn't need a hard per-call cap -- but
# retry/pad on low uniqueness is still real (see sample_and_build_init):
# early-checkpoint LLM output is exactly the kind of low-diversity case
# peptide already had to defend against.
MAX_RETRY_ATTEMPTS = 5


def _run(cmd: list, cwd: Path, cfg: ExperimentConfig | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"+ ({cwd}) {printable}", flush=True)
    # Same interactive-pager-hang risk as peptide's LOLBO family (see
    # peptide_experiment/steps.py::_run's docstring for the full story) --
    # this is the same LOL-BO dependency chain.
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "GIT_PAGER": "cat"}
    if cfg is not None and cfg.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def cleanup_intermediate_epochs(ckpt_dir: Path, final_ckpt: Path) -> None:
    """See peptide_experiment/steps.py's identical helper -- torchtune's
    recipes save a full checkpoint after every epoch with no "final only"
    option."""
    for epoch_dir in ckpt_dir.glob("epoch_*"):
        if epoch_dir.is_dir() and epoch_dir != final_ckpt:
            shutil.rmtree(epoch_dir)


def _write_init_csv(csv_path: Path, xs: list[list[int]], ys: list[float], censoring: list[int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "censoring"])
        for x, y, c in zip(xs, ys, censoring):
            # Comma-joined ints, no brackets/spaces -- the exact schema
            # info_transformer_vae_optimization.py's load_train_data() else-
            # branch (and create_initialization_data.py) already reads/writes.
            writer.writerow([",".join(str(t) for t in x), y, c])


def _pad_to_size(xs: list[list[int]], ys: list[float], censoring: list[int], target_size: int) -> tuple:
    """Known failure mode (mirrors peptide's _pad_to_size): an untuned/
    low-diversity checkpoint can plateau well below target_size unique valid
    candidates no matter how many retries. Pad by resampling with
    replacement rather than crashing -- padding after scoring (not before)
    to avoid wasting real oracle/DB calls on duplicate queries."""
    n = len(xs)
    if n == 0:
        raise RuntimeError("No usable candidates produced")
    if n < target_size:
        rng = random.Random(0)
        pad_idx = [rng.randrange(n) for _ in range(target_size - n)]
        xs = xs + [xs[i] for i in pad_idx]
        ys = ys + [ys[i] for i in pad_idx]
        censoring = censoring + [censoring[i] for i in pad_idx]
        print(f"  padded {n} -> {target_size} candidates by resampling with replacement")
        return xs, ys, censoring
    return xs[:target_size], ys[:target_size], censoring[:target_size]


def sample_and_build_init(
    cfg: ExperimentConfig,
    model_path: Path | str,
    workload: str,
    work_dir: Path,
) -> Path:
    """Sample cfg.init_size candidate query plans for `workload` from the
    checkpoint at `model_path` (via fine-tuning/query_plans/
    sampling_transformer.py), score each valid one in-process with the real
    Postgres oracle (mirrors peptide's steps.py calling apex_wrapper
    directly rather than shelling out again), and write one init CSV ready
    to hand to the BO entry point via --init_csv_path.

    Only used post-first-milestone -- the pre-first-milestone case (no
    fine-tuned checkpoint exists yet) needs no analogous fallback function at
    all: the LOLBO script's own init_w_bao=True default already reads real,
    already-scored BAO plans per-workload, so callers simply don't invoke
    this function for that case (see trajectory_chain.py).
    """
    from your_tasks.your_objective_functions import DatabaseObjective  # noqa: E402 -- relies on config.py's sys.path insert

    work_dir.mkdir(parents=True, exist_ok=True)
    init_csv_path = work_dir / f"{workload}_init.csv"
    if init_csv_path.exists():
        print(f"[{workload}] init data already exists at {init_csv_path}, skipping")
        return init_csv_path

    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR
    # sampling_transformer.py runs in the main env (torch/transformers), same
    # as torchtune -- unlike run_bo()'s LOLBO subprocess below, never
    # cfg.lolbo_python (that's specifically for optimization/query_plans/'s
    # own separate poetry-managed dependency set).
    python = sys.executable

    tasks_csv = work_dir / f"{workload}_sampling_tasks.csv"
    tasks_csv.write_text(f"task\n{workload}\n")

    oracle = DatabaseObjective(
        workload_name=workload,
        worst_runtime_observed=2 * cfg.query_timeout_secs,
        timeout=cfg.query_timeout_secs,
        which_language=cfg.which_query_language,
    )

    n_unique = 0
    seen: set[tuple[int, ...]] = set()
    xs: list[list[int]] = []
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        num_samples = cfg.init_size * attempt
        attempt_jsonl = work_dir / f"{workload}_sampled_attempt{attempt}.jsonl"
        _run(
            [
                python,
                "sampling_transformer.py",
                "--model-path",
                str(model_path),
                "--base-model-path",
                str(cfg.base_checkpoint_dir),
                "--tasks-file",
                str(tasks_csv),
                "--workload-dir",
                str(cfg.bolt_root / WORKLOAD_DIR),
                "--output-file",
                str(attempt_jsonl),
                "--num-samples",
                str(num_samples),
                "--max-tasks",
                "1",
            ],
            cwd=fine_tuning_dir,
            cfg=cfg,
        )
        df = pd.read_json(attempt_jsonl, lines=True)
        row = df[df["task"] == workload].iloc[0]
        for candidate in row["generated_answers"]:
            if not isinstance(candidate, list) or not candidate:
                continue  # unparseable LLM output -- parse_plan() couldn't extract a bracketed list
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            xs.append(list(candidate))

        n_unique = len(xs)
        if n_unique >= cfg.init_size:
            break
        print(
            f"[{workload}] only {n_unique}/{cfg.init_size} unique valid candidates after "
            f"attempt {attempt}/{MAX_RETRY_ATTEMPTS}, sampling {num_samples} more next attempt"
        )
    else:
        print(
            f"[{workload}] giving up on reaching {cfg.init_size} unique candidates after "
            f"{MAX_RETRY_ATTEMPTS} attempts ({n_unique} found); padding instead"
        )

    xs = xs[: cfg.init_size]  # trim any surplus before spending real oracle calls scoring them
    ys, censoring = oracle.query_black_box(xs)
    xs, ys, censoring = _pad_to_size(xs, ys, censoring, cfg.init_size)
    _write_init_csv(init_csv_path, xs, ys, censoring)
    return init_csv_path


def run_bo(
    cfg: ExperimentConfig,
    workload: str,
    work_dir: Path,
    run_id: str,
    init_csv_path: Path | None = None,
) -> Path:
    """Run one single-task BO trial and return the path to its collected-data
    CSV (train_x, train_y, censoring), copied into `work_dir` for permanence.

    init_csv_path=None (default): the pre-first-milestone case, use the
    LOLBO script's own init_w_bao=True default (real, already-scored BAO
    plans per-workload -- no override needed). init_csv_path given: use it
    via the --init_csv_path patch (see
    optimization/query_plans/query_plan_optimization/lolbo_scripts/
    info_transformer_vae_optimization.py's load_train_data()), turning off
    init_w_bao so that patched branch is actually reached.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = work_dir / f"{workload}.csv"
    if dest_csv.exists():
        print(f"[{run_id} {workload}] trajectory already exists at {dest_csv}, skipping")
        return dest_csv

    lolbo_scripts_dir = cfg.bolt_root / LOLBO_SCRIPTS_DIR
    python = cfg.lolbo_python or sys.executable
    wandb_project_name = f"{cfg.experiment_id}_{run_id}"

    cmd = [
        python,
        "info_transformer_vae_optimization.py",
        "--workload_name",
        workload,
        "--track_with_wandb",
        "False",
        "--wandb_project_name",
        wandb_project_name,
        "--max_n_oracle_calls",
        str(cfg.oracle_budget),
        "--bsz",
        str(cfg.bsz),
        "--num_initialization_points",
        str(cfg.init_size),
        "--which_query_language",
        cfg.which_query_language,
        "--allow_cross_joins",
        str(cfg.allow_cross_joins),
    ]
    if cfg.use_pretrained_vae:
        assert cfg.vae_statedict_path, "use_pretrained_vae=True requires vae_statedict_path"
        cmd += ["--path_to_vae_statedict", cfg.vae_statedict_path]
    else:
        cmd += ["--path_to_vae_statedict", ""]
    if init_csv_path is not None:
        cmd += ["--init_w_bao", "False", "--init_csv_path", str(init_csv_path)]
    cmd += ["-", "run_lolbo", "-", "done"]

    _run(cmd, cwd=lolbo_scripts_dir, cfg=cfg)

    produced_csv = (
        lolbo_scripts_dir
        / "optimization_all_collected_data"
        / f"{wandb_project_name}_nowandb_{workload}_all-data-collected.csv"
    )
    if not produced_csv.exists():
        raise RuntimeError(f"Expected BO output at {produced_csv}, but it wasn't produced")
    shutil.copy(produced_csv, dest_csv)
    return dest_csv

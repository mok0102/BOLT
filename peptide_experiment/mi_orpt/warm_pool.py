"""Persistent warm-worker pool for one-step BO calls (paper/method.tex
sec:one-step-pool-evaluation) -- eliminates the ~11s Python-import tax
that `steps.run_bo()`'s subprocess-per-call path re-pays on every single
call (measured via /tmp profiling this session: torch+lolbo_scripts+
uniref_vae imports dominate wall time, the actual VAE-encode/GP-fit/
oracle-score work is only ~2s).

One worker process per GPU, imports everything exactly once at worker
startup, then serves many one-step-BO requests in-process for the rest of
that worker's lifetime -- as opposed to peptide_experiment/mi_orpt/
one_step_evaluator.py's still-used serial fallback, which shells out to
`python info_transformer_vae_optimization.py ...` fresh every call.

Calls `APEXConstrainedDiverseOptimization` directly rather than going
through steps.run_bo()'s CLI/subprocess path -- this also sidesteps that
path's shared `optimization_all_collected_data/`-staging-path collision
class of bug entirely, since results are read directly from the
in-memory LOLBOState rather than a shared intermediate file.

Deliberately no top-level torch/numpy/pandas imports: a spawned worker
must import this module to resolve _init_worker/_run_one_in_worker by
name, so anything imported at module level here runs before
_init_worker's body (including its CUDA_VISIBLE_DEVICES assignment) --
keeping the heavy/CUDA-adjacent imports function-local guarantees they
can't run before that assignment.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import ExperimentConfig
from ..steps import LOLBO_SCRIPTS_DIR

# Per-worker global, set once by _init_worker and reused across every
# work item that worker subsequently processes -- this is the whole
# point: pay the import cost once per worker, not once per call.
_APEX_CLS = None


def _init_worker(gpu_queue, bolt_root: Path) -> None:
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    lolbo_scripts_dir = bolt_root / LOLBO_SCRIPTS_DIR
    optimization_dir = lolbo_scripts_dir.parent
    sys.path.insert(0, str(optimization_dir))
    sys.path.insert(0, str(lolbo_scripts_dir))
    os.chdir(lolbo_scripts_dir)

    global _APEX_CLS
    from info_transformer_vae_optimization import APEXConstrainedDiverseOptimization

    _APEX_CLS = APEXConstrainedDiverseOptimization


def _run_one_in_worker(payload: dict) -> tuple[bool, float | str]:
    """Runs in an already-warmed-up worker process. Returns (True, best_y)
    on success, (False, repr(exc)) on failure -- exceptions aren't allowed
    to propagate raw across the process boundary in a way that's easy to
    read, but the caller still re-raises (matches steps.run_bo()'s
    existing fail-fast behavior -- no new swallowing/retry logic)."""
    import gc

    import numpy as np
    import pandas as pd
    import torch

    try:
        cfg: ExperimentConfig = payload["cfg"]
        task_idx: int = payload["task_idx"]
        pool_seqs: list[str] = payload["pool_seqs"]
        pool_ys: list[float] = payload["pool_ys"]
        work_dir: Path = payload["work_dir"]
        run_id: str = payload["run_id"]
        seed: int = payload["seed"]

        work_dir.mkdir(parents=True, exist_ok=True)
        dest_csv = work_dir / f"task_{task_idx:04d}.csv"
        if dest_csv.exists():
            # Mirrors steps.run_bo()'s own skip-if-exists check -- this
            # path calls the optimizer directly instead of going through
            # run_bo(), so it has to repeat that check itself.
            existing = pd.read_csv(dest_csv)
            return True, float(existing["train_y"].max())

        init_path = work_dir / f"task_{task_idx:04d}_init.txt"
        scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
        init_path.write_text("\n".join(pool_seqs) + "\n")
        scores_path.write_text("\n".join(f"{y:.8f}" for y in pool_ys) + "\n")

        run_name = f"{run_id}_task_{task_idx:04d}"
        opt = _APEX_CLS(
            task_id="apex",
            max_n_oracle_calls=cfg.bsz,
            bsz=cfg.bsz,
            constraint_function_ids=["similarity"],
            constraint_thresholds=[cfg.similarity_threshold],
            constraint_types=[task_idx],
            num_initialization_points=len(pool_seqs),
            init_n_update_epochs=20,
            max_string_length=30,
            task_specific_args=[cfg.task_specific_args],
            init_offset_helper=task_idx,
            track_with_wandb=False,
            wandb_project_name=cfg.experiment_id,
            wandb_run_name=run_name,
            init_data_path=str(init_path),
            init_scores_path=str(scores_path),
            seed=seed,
        )
        opt.run_lolbo()

        best_y = float(opt.lolbo_state.train_y.max())
        # Preserve the same on-disk dest_csv shape steps.run_bo() produces
        # (train_x/train_y columns) -- used repeatedly this session for
        # file-timestamp-based timing diagnostics, and may be relied on by
        # other tooling that inspects these work dirs.
        df = {
            "train_x": np.array(opt.lolbo_state.train_x),
            "train_y": opt.lolbo_state.train_y.squeeze().detach().cpu().numpy(),
        }
        pd.DataFrame.from_dict(df).to_csv(dest_csv, index=None)

        del opt
        gc.collect()
        torch.cuda.empty_cache()
        return True, best_y
    except Exception as e:  # noqa: BLE001 -- reported to caller, not swallowed
        return False, repr(e)


def create_pool(gpus: list[str], bolt_root: Path) -> ProcessPoolExecutor:
    """One long-lived worker process per GPU in `gpus`, each permanently
    claiming exactly one GPU (via the pre-loaded queue) and importing the
    heavy LOLBO stack exactly once. Caller must .shutdown() this when
    done (e.g. in a try/finally around a task loop)."""
    ctx = multiprocessing.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    return ProcessPoolExecutor(
        max_workers=len(gpus),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(gpu_queue, bolt_root),
    )

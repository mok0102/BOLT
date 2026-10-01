"""Persistent warm-worker pool for one-step BO calls, mol domain. Mirrors
peptide_experiment/mi_orpt/warm_pool.py's shape (read-only reference, never
imported -- isolation contract): one worker process per GPU, claims its own
GPU once via a shared queue, then serves many one-step-BO requests for the
rest of its lifetime -- this is what lets construct_pairs_for_task's
[candidate x background] grid of one-step-BO calls actually run across
every GPU in --parallel-gpus concurrently, instead of one_step_evaluator.py's
serial fallback (worker_pool=None) using only the calling process's own
single pinned device.

Differs from peptide's own warm_pool.py in what the pool is FOR: peptide's
own run_bo shells out to a fresh subprocess per call, so that pool's main
payoff is amortizing an ~11s Python-import tax across many calls. mol's
mol_run_bo already runs in-process (mol_experiment/steps.py's own module
docstring), so there's no import tax to amortize -- the payoff here is
purely running independent one-step-BO calls on DIFFERENT GPUs
concurrently, cutting wall-clock by ~len(gpus)x for the (candidate,
background) grid.

Deliberately no top-level torch/mol_lolbo/mol_oracle imports: a spawned
worker must import this module to resolve _init_mol_worker/
_run_one_step_in_worker by name, so anything imported at module level here
runs before _init_mol_worker's body (including its CUDA_VISIBLE_DEVICES
assignment) -- keeping the heavy/CUDA-adjacent imports function-local
guarantees they can't bind the wrong device first (same reasoning as
peptide's own warm_pool.py and this repo's own warm_scoring_pool.py).
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor


def _init_mol_worker(gpu_queue) -> None:
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def _run_one_step_in_worker(payload: dict) -> tuple[bool, float | str]:
    """Runs in an already-GPU-pinned worker process. Returns (True, U1) on
    success, (False, repr(exc)) on failure -- exceptions aren't allowed to
    propagate raw across the process boundary, but the caller still
    re-raises (matches one_step_evaluator.py's own serial-path fail-fast
    behavior -- no new swallowing/retry logic)."""
    try:
        from mol_experiment.mi_orpt.one_step_evaluator import run_one_step_pool

        u1 = run_one_step_pool(
            payload["cfg"],
            payload["task_idx"],
            payload["pool_seqs"],
            payload["pool_ys"],
            payload["work_dir"],
            payload["run_id"],
            payload["seed"],
        )
        return True, u1
    except Exception as e:  # noqa: BLE001 -- reported to caller, not swallowed
        return False, repr(e)


def create_mol_pool(gpus: list[str]) -> ProcessPoolExecutor:
    """One long-lived worker process per GPU in `gpus`, each permanently
    claiming exactly one GPU (via the pre-loaded queue). Caller must
    .shutdown() this when done (e.g. in a try/finally around a task loop) --
    same lifetime convention as warm_scoring_pool.py's pool, shared across
    every task in one build_pairs.py invocation, not recreated per task."""
    ctx = multiprocessing.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    return ProcessPoolExecutor(
        max_workers=len(gpus),
        mp_context=ctx,
        initializer=_init_mol_worker,
        initargs=(gpu_queue,),
    )

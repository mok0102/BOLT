"""Persistent GPU-parallel pool for trajectory_chain.py's per-task dispatch
(build init + full BO trajectory, mol_experiment.trajectory_chain._run_single_task).
Mirrors this repo's own mi_orpt/warm_pool.py / warm_scoring_pool.py /
warm_sampling_pool.py pattern (one worker process per GPU, pinned once via
CUDA_VISIBLE_DEVICES before any torch import) -- lets independent tasks
within one milestone segment (they all sample from the same checkpoint, or
all use mutation-init pre-milestone -- see trajectory_chain.py's
_run_segment single-checkpoint assertion) run concurrently across
cfg.mi_parallel_gpus instead of strictly one at a time.

Motivated by real profiling (2026-09-30): mol_run_bo's own acquisition loop
measured ~1.3s/BO-step (candidate generation ~0.9s dominates; GP update is
cheap after the first step, ~0.02s) at production init_size=1000/
oracle_budget=20000 scale -- ~8 min/task, serial, on ONE GPU, while
cfg.mi_parallel_gpus's other 3 GPUs sat idle for this entire pre-milestone
phase. That per-step cost is intrinsic to the (byte-identical, peptide-shared)
BO engine and candidate generator, not fixable by more parallelism within a
single task's own trajectory -- but tasks ACROSS a segment are fully
independent, so running them concurrently is the correct lever, not
shaving the per-step cost further.

Deliberately no top-level torch/mol imports -- see mi_orpt/warm_pool.py's
own docstring for why (a spawned worker importing this module to resolve
_init_task_worker/_run_task_in_worker by name must not trigger CUDA binding
before the initializer's CUDA_VISIBLE_DEVICES assignment runs).

Simpler than mi_orpt/warm_sampling_pool.py: does not pre-load an LLM
checkpoint per worker. Once checkpoint_to_sample_from() resolves to a real
model (milestone reached) and mol_sample_and_build_init runs inside a
worker here, that worker still loads the checkpoint fresh for every task
dispatched to it -- correct, but not warm-cached the way build_orpt_pairs'
own use of warm_sampling_pool.py is. Worth revisiting if per-task reload
cost becomes dominant post-milestone the way the unparallelized acquisition
loop was pre-milestone.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor


def _init_task_worker(gpu_queue) -> None:
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def _run_task_in_worker(payload: dict) -> tuple[bool, str]:
    """Runs in an already-GPU-pinned worker process. Returns (True, "") on
    success, (False, repr(exc)) on failure -- exceptions aren't allowed to
    propagate raw across the process boundary, but the caller still
    re-raises (no new swallowing/retry logic)."""
    try:
        from mol_experiment.trajectory_chain import _run_single_task

        _run_single_task(payload["cfg"], payload["task_idx"])
        return True, ""
    except Exception as e:  # noqa: BLE001 -- reported to caller, not swallowed
        return False, repr(e)


def create_task_pool(gpus: list[str]) -> ProcessPoolExecutor:
    """One long-lived worker process per GPU in `gpus`, each permanently
    claiming exactly one GPU. Caller must .shutdown() this when done (e.g.
    in a try/finally around a segment) -- same lifetime convention as this
    repo's other warm pools, shared across every task in one segment, not
    recreated per task."""
    ctx = multiprocessing.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    return ProcessPoolExecutor(
        max_workers=len(gpus),
        mp_context=ctx,
        initializer=_init_task_worker,
        initargs=(gpu_queue,),
    )

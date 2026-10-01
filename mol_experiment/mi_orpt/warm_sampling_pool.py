"""Persistent GPU-parallel pool for mol_sample_and_build_init's LLM
generation step. Mirrors this repo's own warm_scoring_pool.py exactly (one
worker process per GPU, model loaded once per worker and reused across
every task subsequently dispatched to it) and peptide's own
mi_orpt/warm_sampling_pool.py's motivation: mol_experiment/steps.py's
mol_sample_and_build_init reloads the full checkpoint from scratch on every
task otherwise -- real, measured overhead (peptide's own analogous number:
~31s load for ~7s of generation) that gets paid once per task per milestone
via cfg.mi_candidate_temperature's dedicated-pool sampling
(mol_experiment/orpt.py::build_orpt_pairs), across every task in
[0, milestone) at production scale.

Deliberately a separate pool from warm_pool.py's one-step-BO workers and
warm_scoring_pool.py's reference-scoring workers (different loaded model,
different lifetime granularity) -- same reasoning as warm_scoring_pool.py's
own docstring: keeps each pool's failure modes independent.

Deliberately no top-level torch/mol_sampling imports: a spawned worker must
import this module to resolve _init_sampling_worker/_sample_task_in_worker
by name, so anything imported at module level here runs before
_init_sampling_worker's body (including its CUDA_VISIBLE_DEVICES
assignment) -- keeping the heavy/CUDA-adjacent imports function-local
guarantees they can't bind the wrong device first.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Per-worker globals, set once by _init_sampling_worker and reused across
# every task that worker subsequently samples.
_MODEL = None
_TOKENIZER = None
_DEVICE = None


def _init_sampling_worker(gpu_queue, model_path: Path) -> None:
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    global _MODEL, _TOKENIZER, _DEVICE
    from mol_sampling import load_model_and_tokenizer

    _MODEL, _TOKENIZER, _DEVICE = load_model_and_tokenizer(model_path)


def _sample_task_in_worker(payload: dict) -> tuple[bool, tuple[str, str] | str]:
    """Runs in an already-GPU-pinned, already-model-loaded worker process.
    Returns (True, (init_path_str, scores_path_str)) on success, (False,
    repr(exc)) on failure -- exceptions aren't allowed to propagate raw
    across the process boundary, but the caller still re-raises (no new
    swallowing/retry logic)."""
    try:
        from mol_experiment.steps import mol_sample_and_build_init

        init_path, scores_path = mol_sample_and_build_init(
            payload["cfg"],
            payload["model_path"],
            payload["task_idx"],
            payload["work_dir"],
            payload["temperature"],
            preloaded_model=(_MODEL, _TOKENIZER, _DEVICE),
        )
        return True, (str(init_path), str(scores_path))
    except Exception as e:  # noqa: BLE001 -- reported to caller, not swallowed
        return False, repr(e)


def create_sampling_pool(gpus: list[str], model_path: Path) -> ProcessPoolExecutor:
    """One long-lived worker process per GPU in `gpus`, each permanently
    claiming exactly one GPU and loading model_path exactly once. Caller
    must .shutdown() this when done (e.g. in a try/finally around a task
    loop) -- same lifetime convention as warm_pool.py/warm_scoring_pool.py's
    own pools, shared across every task in one build_orpt_pairs() milestone,
    not recreated per task."""
    ctx = multiprocessing.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    return ProcessPoolExecutor(
        max_workers=len(gpus),
        mp_context=ctx,
        initializer=_init_sampling_worker,
        initargs=(gpu_queue, model_path),
    )


def sample_tasks_parallel(
    pool: ProcessPoolExecutor,
    cfg,
    model_path: Path,
    task_indices: list[int],
    work_dir: Path,
    temperature: float | None,
) -> dict[int, tuple[Path, Path]]:
    """Dispatches every task_idx in task_indices across pool's workers,
    returning {task_idx: (init_path, scores_path)}."""
    futures = {}
    for task_idx in task_indices:
        payload = {
            "cfg": cfg,
            "model_path": model_path,
            "task_idx": task_idx,
            "work_dir": work_dir,
            "temperature": temperature,
        }
        futures[pool.submit(_sample_task_in_worker, payload)] = task_idx

    results: dict[int, tuple[Path, Path]] = {}
    for future in futures:
        task_idx = futures[future]
        ok, value = future.result()
        if not ok:
            raise RuntimeError(f"mol_sample_and_build_init worker failed for task {task_idx}: {value}")
        init_path_str, scores_path_str = value
        results[task_idx] = (Path(init_path_str), Path(scores_path_str))
    return results

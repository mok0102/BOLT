"""Persistent GPU-parallel pool for likelihood.py::score_sequences() --
same warm-worker pattern as warm_pool.py, applied to the reference-model
scoring step used for matched-intervention ORPT's reference-aligned
candidate distribution q_t.

Measured this session: score_sequences() previously reloaded the full
milestone checkpoint from scratch on every task (~3.25s wasted per task,
same checkpoint every time within one build_pairs.py invocation) and
always ran on a single GPU regardless of how many were configured via
mi_parallel_gpus -- confirmed via nvidia-smi during a live run: 55+
seconds of continuous single-GPU activity while every other configured
GPU sat completely idle. Bank sizes observed ranged up to ~11,600
candidates at a measured ~6.5ms/sequence, i.e. up to ~75s of scoring
alone, per task, on one GPU. This pool fixes both: the model loads once
per pool (i.e. once per milestone, shared across every task), and
scoring work is chunked across every GPU in the pool concurrently.

Deliberately a separate pool from warm_pool.py's one-step-BO workers
(different loaded model, different lifetime granularity within
build_pairs.py's per-task loop) rather than extending that
already-verified pool to do double duty -- keeps each pool's failure
modes independent and avoids touching tested code for an unrelated
concern.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Per-worker globals, set once by _init_scoring_worker and reused across
# every chunk that worker subsequently scores.
_MODEL = None
_TOKENIZER = None


def _init_scoring_worker(gpu_queue, torchtune_config_path: Path, checkpoint_dir: Path) -> None:
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    global _MODEL, _TOKENIZER
    from .likelihood import load_model_and_tokenizer

    _MODEL, _TOKENIZER = load_model_and_tokenizer(torchtune_config_path, checkpoint_dir)


def _score_chunk_in_worker(payload: dict) -> tuple[bool, list[float] | str]:
    try:
        from .likelihood import score_sequences

        log_probs = score_sequences(
            _MODEL,
            _TOKENIZER,
            context=payload["context"],
            sequences=payload["sequences"],
            system_prompt=payload["system_prompt"],
            batch_size=payload["batch_size"],
        )
        return True, log_probs
    except Exception as e:  # noqa: BLE001 -- reported to caller, not swallowed
        return False, repr(e)


def create_scoring_pool(gpus: list[str], torchtune_config_path: Path, checkpoint_dir: Path) -> ProcessPoolExecutor:
    """One long-lived worker process per GPU in `gpus`, each permanently
    claiming exactly one GPU and loading the scoring model exactly once.
    Caller must .shutdown() this when done (e.g. in a try/finally around a
    task loop) -- same lifetime convention as mi_orpt/warm_pool.py's pool,
    shared across every task in one build_pairs.py invocation, not
    recreated per task."""
    ctx = multiprocessing.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    return ProcessPoolExecutor(
        max_workers=len(gpus),
        mp_context=ctx,
        initializer=_init_scoring_worker,
        initargs=(gpu_queue, torchtune_config_path, checkpoint_dir),
    )


def score_sequences_parallel(
    pool: ProcessPoolExecutor,
    num_workers: int,
    context: str,
    sequences: list[str],
    system_prompt: str,
    batch_size: int = 32,
) -> list[float]:
    """Splits `sequences` into num_workers contiguous chunks, scores them
    concurrently across the pool, and reassembles the result in the exact
    same order as the input (callers like build_pairs.py zip log_probs
    positionally against the bank, so order must be preserved)."""
    if not sequences:
        return []
    num_chunks = max(1, min(num_workers, len(sequences)))
    chunk_size = -(-len(sequences) // num_chunks)  # ceil div
    chunks = [sequences[i : i + chunk_size] for i in range(0, len(sequences), chunk_size)]

    futures = {}
    for chunk_idx, chunk in enumerate(chunks):
        payload = {
            "context": context,
            "sequences": chunk,
            "system_prompt": system_prompt,
            "batch_size": batch_size,
        }
        futures[pool.submit(_score_chunk_in_worker, payload)] = chunk_idx

    results: list[list[float] | None] = [None] * len(chunks)
    for future in futures:
        chunk_idx = futures[future]
        ok, value = future.result()
        if not ok:
            raise RuntimeError(f"score_sequences chunk {chunk_idx} failed: {value}")
        results[chunk_idx] = value

    log_probs: list[float] = []
    for chunk_result in results:
        log_probs.extend(chunk_result)
    return log_probs

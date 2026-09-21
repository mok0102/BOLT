"""Persistent warm-worker pool for LLM candidate sampling -- the third
instance of warm_pool.py's template (the second is warm_scoring_pool.py).

Motivation, measured on a live run (45s window, 15 snapshots, 8 workers):
sampling subprocesses occupied ~6.5 of 8 concurrent worker slots at ~38s per
call, of which only ~7s was actual generation (100 samples; 29s at 2000). The
other ~31s is `import torch` + CUDA init + `from_pretrained` + teardown, re-paid
for EVERY task because steps.sample_and_build_init() shells out to
fine-tuning/peptides/sampling_transformers.py once per task. Checkpoint shard
reads are not the cost (~0s, page-cached) -- it's process startup. Roughly half
of total pipeline wall-clock was going into reloading the same 3B checkpoint.

One worker process per GPU loads the checkpoint exactly once and then serves
many per-task generate requests for the rest of that worker's lifetime.

ONE POOL PER GPU, not one shared pool with N workers. ProcessPoolExecutor
cannot route a job to a specific worker, so a shared pool would let two
2000-way `model.generate` calls (mi_candidate_pool_size) plus a BO subprocess
plus an APEX-oracle subprocess all land on one card -- a per-GPU memory profile
nobody has measured. Per-GPU pools keep today's per-card workload identical
apart from the ~6GB resident checkpoint, preserve trajectory_chain.py's
"permanently pinned thread, not round-robin" invariant, and contain failures:
a ProcessPoolExecutor is permanently BrokenProcessPool once any worker dies, so
with a shared pool a single transient OOM would kill sampling for the remainder
of a 200-task (multi-day) segment.

Separate pool per loaded model, per warm_scoring_pool.py's stated convention --
_generate_in_worker asserts the job's model_path matches the one baked into the
worker, because a stale pool would otherwise emit silently wrong samples.

Two hard rules:

1. No top-level torch / transformers / peptide_experiment imports. A spawned
   worker re-imports this module to resolve _init_sampling_worker and
   _generate_in_worker by name, and that import runs BEFORE
   _init_sampling_worker's body (including its CUDA_VISIBLE_DEVICES
   assignment). warm_pool.py imports ..config at module level and gets away
   with it only because apex_oracle was separately fixed to load models lazily
   to CPU; this module does not rely on that luck. Module-level imports are
   stdlib only, and the job payload is plain primitives rather than an
   ExperimentConfig so the worker never has to import peptide_experiment.config
   (which pulls in torch + apex_oracle on every re-import).

2. mp_context="spawn" is mandatory, not stylistic. By the time a pool is
   created the driver process already holds a live CUDA context, because
   steps.build_mutation_init / steps._ensure_constraint_feasible call
   apex_wrapper in-process. fork would hand that context to children.

The CLI path in steps.sample_and_build_init() is deliberately kept intact: it
is both the no-GPU/serial fallback and the degradation path when a pool breaks.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

# Per-worker globals, set once by _init_sampling_worker and reused by every
# job that worker subsequently handles -- the entire point of this module.
_MODEL = None
_TOKENIZER = None
_DEVICE = None
_MODEL_PATH = None


def _init_sampling_worker(gpu_queue, fine_tuning_dir: Path, model_path: Path) -> None:
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    sys.path.insert(0, str(fine_tuning_dir))

    global _MODEL, _TOKENIZER, _DEVICE, _MODEL_PATH
    from sampling_transformers import load_model_and_tokenizer

    _MODEL, _TOKENIZER, _DEVICE = load_model_and_tokenizer(model_path)
    _MODEL_PATH = Path(model_path).expanduser().resolve()


def _generate_in_worker(payload: dict) -> tuple[bool, str]:
    """Runs in an already-warmed worker. Returns (True, output_file) or
    (False, repr(exc)) -- exceptions are reported to the caller rather than
    propagated raw across the process boundary, matching warm_pool.py."""
    import gc

    import torch

    try:
        from sampling_transformers import (
            REFERENCE_SEQUENCE,
            generate_for_reference,
            write_records,
        )

        requested = Path(payload["model_path"]).expanduser().resolve()
        if requested != _MODEL_PATH:
            # A pool outliving its checkpoint would produce samples from the
            # wrong model with no other symptom, so fail loudly instead.
            return False, (
                f"sampling pool holds {_MODEL_PATH}, but job requested {requested} -- "
                "a pool is valid for exactly one checkpoint"
            )

        task_idx: int = payload["task_idx"]
        if not 0 <= task_idx < len(REFERENCE_SEQUENCE):
            return False, f"task_idx {task_idx} out of range for {len(REFERENCE_SEQUENCE)} references"
        peptide = REFERENCE_SEQUENCE[task_idx]

        answers = generate_for_reference(
            _MODEL,
            _TOKENIZER,
            peptide,
            device=_DEVICE,
            temperature=payload["temperature"],
            top_p=payload["top_p"],
            max_new_tokens=payload["max_new_tokens"],
            num_samples=payload["samples_per_peptide"],
        )
        write_records(payload["output_file"], [{"source_peptide": peptide, "generated_answers": answers}])
        return True, str(payload["output_file"])
    except Exception as e:  # noqa: BLE001 -- reported to caller, not swallowed
        return False, repr(e)
    finally:
        # The generate arena (up to MAX_SAMPLES_PER_CALL return sequences) used
        # to be reclaimed by process exit; in a persistent worker it isn't.
        gc.collect()
        torch.cuda.empty_cache()


class SamplingPool:
    """One GPU's single-worker pool, with the checkpoint it was built for.

    `broken` latches once the underlying executor dies so the caller can stop
    retrying and fall back to the CLI path for the rest of the segment.
    """

    def __init__(self, executor: ProcessPoolExecutor, gpu: str, model_path: Path):
        self.executor = executor
        self.gpu = gpu
        self.model_path = model_path
        self.broken = False

    def generate(
        self,
        *,
        task_idx: int,
        samples_per_peptide: int,
        output_file: Path,
        temperature: float | None = None,
        top_p: float = 0.95,
        max_new_tokens: int = 64,
    ) -> None:
        """Blocks until this GPU's worker has written `output_file`.

        Raises RuntimeError on a job-level failure, or lets BrokenProcessPool
        propagate (after latching `broken`) so the caller can fall back.
        """
        payload = {
            "task_idx": task_idx,
            "samples_per_peptide": samples_per_peptide,
            # sampling_transformers.py's own CLI defaults, so the pooled path
            # and the CLI path generate under identical settings.
            "temperature": 1 if temperature is None else temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "output_file": Path(output_file),
            "model_path": Path(self.model_path),
        }
        try:
            ok, value = self.executor.submit(_generate_in_worker, payload).result()
        except BrokenProcessPool:
            self.broken = True
            raise
        if not ok:
            raise RuntimeError(f"warm sampling pool (gpu {self.gpu}) failed on task {task_idx}: {value}")


def create_sampling_pools(gpus: list[str], fine_tuning_dir: Path, model_path: Path) -> dict[str, SamplingPool]:
    """One single-worker pool per GPU, each permanently pinned to that GPU and
    holding `model_path` resident. Caller MUST shut these down before any
    `tune run` starts -- SFT peaks near 27GB/GPU and DPO near 36GB, and 36 + the
    ~6GB resident checkpoint does not fit on a 40GB card. See
    shutdown_sampling_pools and the try/finally at both call sites.
    """
    ctx = multiprocessing.get_context("spawn")
    pools: dict[str, SamplingPool] = {}
    for gpu in gpus:
        gpu_queue = ctx.Queue()
        gpu_queue.put(gpu)
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=_init_sampling_worker,
            initargs=(gpu_queue, fine_tuning_dir, model_path),
        )
        pools[gpu] = SamplingPool(executor, gpu, model_path)
    print(f"[warm sampling pool] {len(pools)} worker(s) on GPU(s) {','.join(gpus)} holding {model_path}")
    return pools


def shutdown_sampling_pools(pools: dict[str, SamplingPool]) -> None:
    for pool in pools.values():
        pool.executor.shutdown(wait=True)
    if pools:
        print(f"[warm sampling pool] released {len(pools)} worker(s)")

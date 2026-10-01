"""Persistent CPU-only pool for mol_init_candidates.generate_feasible_candidates_around_seed.

Real profiling (2026-09-30, mol_experiment production run) established two facts that motivate
this module's shape:

1. Generation cost scales ~linearly with n_feasible (measured 10/25/50/100 ->
   0.053/0.110/0.198/0.383s, ~6.1-7.7 attempts/feasible throughout) -- this is genuine CPU
   compute (RDKit decode + Tanimoto filtering in a rejection-sampling loop), not oracle-style
   fixed-per-call overhead. So splitting one call's work across workers is a real lever, not an
   illusory one.
2. A FRESH multiprocessing.Pool per call is a large net loss: measured 4.2-4.4s wall time for a
   4-way pool spawned per call (process spawn + rdkit/selfies reimport + SELFIES vocab reload
   dominates), vs 0.75-0.8s serial at the real production n_candidates=200. A persistent/warm
   pool -- created once per task in mol_experiment/steps.py::mol_run_bo, reused across every BO
   step in that task -- is required, mirroring this repo's existing warm_pool.py/
   warm_scoring_pool.py/warm_sampling_pool.py/warm_task_pool.py pattern.

Unlike those other warm pools, this one is CPU-only: no GPU queue, no CUDA_VISIBLE_DEVICES
pinning, so the initializer is trivial. Workers still pay the SELFIES vocab load
(mol_init_candidates._vocab_tokens/_vocab_set, process-level globals) once each, on their first
real chunk -- negligible amortized over a ~8-9min task.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def _generate_chunk_in_worker(payload: dict) -> tuple[bool, tuple[list, dict] | str]:
    """Runs in a plain (unpinned) worker process. Returns (True, (candidates, stats)) on
    success, (False, repr(exc)) on failure -- exceptions aren't allowed to propagate raw across
    the process boundary, but the caller still re-raises (no new swallowing/retry logic)."""
    try:
        from mol_init_candidates import generate_feasible_candidates_around_seed

        cands, stats = generate_feasible_candidates_around_seed(
            payload["seed_smiles"],
            payload["tau_mol"],
            payload["n_feasible"],
            payload["rng_seed"],
        )
        return True, (cands, stats)
    except Exception as e:  # noqa: BLE001 -- reported to caller, not swallowed
        return False, repr(e)


def create_generation_pool(n_workers: int) -> ProcessPoolExecutor:
    """n_workers plain worker processes, no GPU pinning. Caller must .shutdown() this when done
    (e.g. in a try/finally around a task's BO loop) -- same lifetime convention as this repo's
    other warm pools: created once per task, shared across every BO step in that task, not
    recreated per step."""
    ctx = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)


def generate_feasible_candidates_parallel(
    pool: ProcessPoolExecutor,
    seed_smiles: str,
    tau_mol: float,
    n_feasible: int,
    rng_seed: int,
    n_workers: int,
) -> tuple[list[tuple[str, str]], dict]:
    """Splits n_feasible across n_workers chunks (each with a distinct deterministic rng_seed
    offset), runs them concurrently on `pool`, and merges the results.

    Each worker dedupes only within its own chunk (its own local `seen` set, seeded from the
    same seed molecule) -- so, unlike a single serial call, cross-worker duplicate candidates
    are possible. This function dedupes the merged output by canonical SMILES to catch them.
    The merged pool may therefore be marginally smaller than a serial call's n_feasible in rare
    collision cases -- acceptable, since mol_acquisition_step already does
    min(bsz, len(cand_smiles)) downstream."""
    chunk_size = -(-n_feasible // n_workers)  # ceil division
    futures = []
    for i in range(n_workers):
        payload = {
            "seed_smiles": seed_smiles,
            "tau_mol": tau_mol,
            "n_feasible": chunk_size,
            "rng_seed": rng_seed * 1_000_003 + i,  # distinct, deterministic per chunk
        }
        futures.append(pool.submit(_generate_chunk_in_worker, payload))

    seen: set[str] = set()
    merged: list[tuple[str, str]] = []
    total_attempts = 0
    total_valid_molecule = 0
    for future in futures:
        ok, value = future.result()
        if not ok:
            raise RuntimeError(f"generate_feasible_candidates_around_seed worker failed: {value}")
        cands, stats = value
        total_attempts += stats["n_attempts"]
        total_valid_molecule += stats["n_valid_molecule"]
        for canon, selfies_str in cands:
            if canon in seen:
                continue
            seen.add(canon)
            merged.append((canon, selfies_str))

    merged_stats = {
        "n_attempts": total_attempts,
        "n_valid_molecule": total_valid_molecule,
        "n_feasible_found": len(merged),
        "yield_feasible_per_attempt": len(merged) / total_attempts if total_attempts else 0.0,
    }
    return merged, merged_stats

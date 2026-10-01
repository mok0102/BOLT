"""Shared low-level steps for one mol task: build init candidates, and run one
BO trial. Mirrors peptide_experiment/steps.py's shape (read-only reference, never
imported -- isolation contract).

Domain-generic helpers below (_run, materialize_hf_checkpoint,
cleanup_intermediate_epochs, distributed_finetune_launch, _pad_to_size) are
copied byte-for-byte from peptide_experiment/steps.py per the spec's "Code
layout: Domain-generic code is copied" section (verified zero peptide/apex/amino
content in any of them) -- only the ExperimentConfig type hint changed to
MolExperimentConfig. mol_ensure_constraint_feasible / mol_build_mutation_init /
mol_run_bo / mol_sample_and_build_init are new, mol-specific implementations
(peptide's equivalents shell out to a CLI subprocess wrapping LOLBO's Optimize
class or sampling_transformers.py; mol runs the same Milestone-4-verified BO
machinery -- LOLBOState + MoleculeObjective + mol_acquisition_step -- and, for
sampling, optimization/mol/mol_sampling.py's HF-transformers generation, both
in-process instead of via subprocess -- a deliberate skeleton-stage
simplification, not a port of peptide's subprocess-CLI pattern).

mol_sample_and_build_init (LLM-sampling analogue of peptide's
sample_and_build_init, used once a fine-tuned checkpoint exists at
milestone > 0) is implemented below. Simplifications vs. peptide's version,
explicit rather than silent:
- No warm sampling pool / subprocess-per-attempt split (peptide's
  mi_orpt/warm_sampling_pool.py): generation runs in-process here, loading the
  checkpoint fresh once per task -- same skeleton-stage choice as mol_run_bo.
  Add a mol warm-sampling-pool equivalent if/when parallel multi-GPU task
  dispatch is wired into trajectory_chain.py (see that module's docstring).
- No temperature_step widening path -- only a plain `temperature` override
  (falls back to 1.0 when None), passed in by the caller rather than read
  from cfg directly (see mol_sample_and_build_init's own docstring for why).
"""

from __future__ import annotations

import dataclasses
import fcntl
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from mol_experiment.config import MolExperimentConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimization" / "mol"))
FINE_TUNING_DIR = "fine-tuning/mol"


def _run(cmd: list, cwd: Path, cfg: MolExperimentConfig | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"+ ({cwd}) {printable}", flush=True)
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


def materialize_hf_checkpoint(ckpt_dir: Path, epoch: int, base_checkpoint_dir: Path) -> Path:
    """Byte-identical port of peptide_experiment/steps.py::materialize_hf_checkpoint
    -- see that docstring for the full torchtune-checkpoint-format rationale."""
    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        return ckpt_dir

    epoch_dir = ckpt_dir / f"epoch_{epoch}"
    if (epoch_dir / "model.safetensors.index.json").exists():
        for item in list(epoch_dir.iterdir()):
            is_stray_epoch_dir = item.is_dir() and item.name.startswith("epoch_") and item.name[len("epoch_"):].isdigit()
            if item.is_dir() and (item.name == "recipe_state" or is_stray_epoch_dir):
                shutil.rmtree(item)
                continue
            shutil.move(str(item), str(ckpt_dir / item.name))
        epoch_dir.rmdir()
        return ckpt_dir

    from safetensors.torch import save_file

    shards = sorted(ckpt_dir.glob(f"hf_model_*_{epoch}.pt"))
    if not shards:
        raise FileNotFoundError(
            f"materialize_hf_checkpoint: no hf_model_*_{epoch}.pt shards in {ckpt_dir} "
            "(expected torchtune's FullModelHFCheckpointer to have written them)"
        )
    num_shards = len(shards)
    for cpt_idx, shard_path in enumerate(shards, start=1):
        state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
        seen_ptrs: dict[int, str] = {}
        for key, tensor in state_dict.items():
            ptr = tensor.data_ptr()
            if ptr in seen_ptrs:
                state_dict[key] = tensor.clone()
            else:
                seen_ptrs[ptr] = key
        out_path = ckpt_dir / f"model-{cpt_idx:05d}-of-{num_shards:05d}.safetensors"
        save_file(state_dict, out_path, metadata={"format": "pt"})

    base_index = base_checkpoint_dir / "model.safetensors.index.json"
    if not base_index.exists():
        raise FileNotFoundError(
            f"materialize_hf_checkpoint: base checkpoint has no model.safetensors.index.json at {base_index}"
        )
    shutil.copy2(base_index, index_path)

    for tokenizer_file in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"):
        src = base_checkpoint_dir / tokenizer_file
        dst = ckpt_dir / tokenizer_file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return ckpt_dir


def cleanup_intermediate_epochs(ckpt_dir: Path, final_epoch: int) -> None:
    """Byte-identical port of peptide_experiment/steps.py::cleanup_intermediate_epochs."""
    materialized = (ckpt_dir / "model.safetensors.index.json").exists()
    for pattern in ("hf_model_*_*.pt", "adapter_*.pt"):
        for f in ckpt_dir.glob(pattern):
            try:
                epoch = int(f.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if epoch != final_epoch or materialized:
                f.unlink()
    recipe_state = ckpt_dir / "recipe_state.pt"
    if recipe_state.exists():
        recipe_state.unlink()
    for d in ckpt_dir.glob("epoch_*"):
        if not d.is_dir():
            continue
        suffix = d.name[len("epoch_"):]
        if not suffix.isdigit():
            continue
        epoch = int(suffix)
        if epoch != final_epoch or materialized:
            shutil.rmtree(d)
    recipe_state_dir = ckpt_dir / "recipe_state"
    if recipe_state_dir.is_dir():
        shutil.rmtree(recipe_state_dir)


def distributed_finetune_launch(
    cfg: MolExperimentConfig, torchtune_config_name: str, fine_tuning_dir: Path
) -> tuple[list[str], list[str], MolExperimentConfig]:
    """Byte-identical port of peptide_experiment/steps.py::distributed_finetune_launch
    -- see that docstring for the master-port-collision and per-device-batch-size
    rationale."""
    if not cfg.mi_parallel_gpus:
        return [], [], cfg

    nproc_per_node = len(cfg.mi_parallel_gpus)
    master_port = 29500 + int(cfg.mi_parallel_gpus[0])
    with open(fine_tuning_dir / "torchtune_config" / torchtune_config_name) as f:
        per_gpu_batch_size = yaml.safe_load(f)["batch_size"]
    print(
        f"[distributed finetune] {torchtune_config_name}: nproc_per_node={nproc_per_node}, "
        f"batch_size={per_gpu_batch_size}/device "
        f"(effective global batch size {per_gpu_batch_size * nproc_per_node}), "
        f"master_port={master_port}"
    )

    launch_cfg = dataclasses.replace(cfg, cuda_visible_devices=",".join(cfg.mi_parallel_gpus))
    return ["--master-port", str(master_port)], [f"batch_size={per_gpu_batch_size}"], launch_cfg


def _pad_to_size(init_path: Path, scores_path: Path, target_size: int) -> None:
    """Byte-identical port of peptide_experiment/steps.py::_pad_to_size."""
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [line for line in scores_path.read_text().splitlines() if line.strip()]
    n = min(len(seqs), len(scores))
    if n == 0:
        raise RuntimeError(f"No usable candidates produced in {init_path}")
    seqs, scores = seqs[:n], scores[:n]

    if n < target_size:
        rng = random.Random(0)
        pad_idx = [rng.randrange(n) for _ in range(target_size - n)]
        seqs = seqs + [seqs[i] for i in pad_idx]
        scores = scores + [scores[i] for i in pad_idx]
        print(f"  padded {n} -> {target_size} candidates by resampling with replacement")
    else:
        seqs, scores = seqs[:target_size], scores[:target_size]

    init_path.write_text("\n".join(seqs) + "\n")
    scores_path.write_text("\n".join(scores) + "\n")


def mol_build_mutation_init(cfg: MolExperimentConfig, task_idx: int, work_dir: Path) -> tuple[Path, Path]:
    """Mol analogue of peptide_experiment/steps.py::build_mutation_init: build init
    candidates for a task via bounded, Tanimoto-feasible SELFIES edits around the
    seed (mol_init_candidates.py::generate_feasible_candidates_around_seed,
    verified in Milestone 4 to produce a real, non-degenerate init pool with
    measurable downstream BO headroom), scored by the real oracle. Used for every
    task in this skeleton regardless of milestone (peptide only uses its mutation
    fallback pre-milestone-0; mol's LLM-sampling path is still a TODO -- see this
    module's docstring).

    Deliberately does NOT reuse peptide's `max_mutation_distance = 1.0 -
    similarity_threshold` parameterization -- has no molecular analogue (spec:
    "The parameterization does not carry over").
    """
    from mol_init_candidates import generate_feasible_candidates_around_seed
    from mol_oracle import EnsembleMolOracle
    from mol_tasks import get_task

    work_dir.mkdir(parents=True, exist_ok=True)
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    lock_path = work_dir / f"task_{task_idx:04d}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if init_path.exists() and scores_path.exists():
                print(f"[task {task_idx}] init data already exists at {init_path}, skipping")
                return init_path, scores_path

            task = get_task(task_idx)
            cands, gen_stats = generate_feasible_candidates_around_seed(
                task.seed_smiles, tau_mol=cfg.tau_mol, n_feasible=cfg.init_size, rng_seed=task_idx
            )
            print(f"[task {task_idx}] init candidate generation: {gen_stats}")
            smiles_list = [task.seed_smiles] + [c for c, _ in cands]

            oracle = EnsembleMolOracle(target_id=task.target_id, target_sequence=task.sequence)
            scores = oracle.query_oracle(smiles_list, input_kind="smiles")
            keep = [i for i, s in enumerate(scores) if s == s]  # drop NaN
            smiles_list = [smiles_list[i] for i in keep]
            scores = [scores[i] for i in keep]

            init_path.write_text("\n".join(smiles_list) + "\n")
            scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")
            _pad_to_size(init_path, scores_path, cfg.init_size)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return init_path, scores_path


def mol_run_bo(
    cfg: MolExperimentConfig,
    task_idx: int,
    work_dir: Path,
    run_id: str,
    init_path: Path,
    scores_path: Path,
    n_bo_steps: int | None = None,
    seed: int | None = None,
) -> Path:
    """Mol analogue of peptide_experiment/steps.py::run_bo. Peptide shells out to
    a CLI subprocess wrapping LOLBO's Optimize class; this runs the same
    Milestone-4-verified machinery in-process instead (LOLBOState +
    MoleculeObjective + mol_acquisition_step) -- a deliberate skeleton-stage
    simplification (no process isolation between trials, no GPU-pinning via a
    subprocess env), not yet a full port of peptide's CLI-driven pattern.

    Output format matches peptide's own collected-data CSV shape closely enough
    for downstream consumers (train_x, train_y columns) -- feasibility masking
    for reporting is the caller's job (mirrors best_objective_at_k), not
    something this function decides, exactly as in peptide.
    """
    import csv

    from mol_lolbo.lolbo import LOLBOState
    from mol_acquisition import mol_acquisition_step
    from mol_objective import MoleculeObjective
    from mol_oracle import EnsembleMolOracle
    from mol_tasks import get_task

    work_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = work_dir / f"task_{task_idx:04d}.csv"
    if dest_csv.exists():
        print(f"[{run_id} task {task_idx}] trajectory already exists at {dest_csv}, skipping")
        return dest_csv

    task = get_task(task_idx)
    n_bo_steps = n_bo_steps if n_bo_steps is not None else max(1, (cfg.oracle_budget - cfg.init_size) // cfg.bsz)

    oracle = EnsembleMolOracle(target_id=task.target_id, target_sequence=task.sequence)
    objective = MoleculeObjective(oracle=oracle, seed_smiles=task.seed_smiles, task_id=task.target_id)

    init_smiles = [line for line in init_path.read_text().splitlines() if line.strip()]
    init_scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]

    # Defense-in-depth final gate before vae_forward (mol_init_candidates.filter_vocab_safe's
    # own docstring has the full incident this guards against) -- should drop nothing now that
    # the actual upstream bug is fixed, but a silent regression in any future candidate source
    # must not be able to crash vae_forward again.
    from mol_init_candidates import filter_vocab_safe

    vocab_mask = filter_vocab_safe(init_smiles)
    if not all(vocab_mask):
        n_dropped = len(vocab_mask) - sum(vocab_mask)
        print(f"[{run_id} task {task_idx}] dropping {n_dropped}/{len(vocab_mask)} init candidates "
              "that fail the final vocab safety check (should be rare -- report this if it isn't)")
        init_smiles = [s for s, keep in zip(init_smiles, vocab_mask) if keep]
        init_scores = [s for s, keep in zip(init_scores, vocab_mask) if keep]
        if not init_smiles:
            raise RuntimeError(f"task {task_idx}: every init candidate failed the vocab safety check")

    objective.xs_to_scores_dict = dict(zip(init_smiles, init_scores))
    init_z, _ = objective.vae_forward(init_smiles)
    init_z = init_z.detach().cpu()
    init_y = torch.tensor(init_scores, dtype=torch.float32).unsqueeze(-1)

    lolbo_state = LOLBOState(
        objective=objective,
        surrogate_type="gp_dkl",
        train_x=init_smiles,
        train_y=init_y,
        train_z=init_z,
        train_c=None,  # unconstrained latent BO -- see MOL_LATENT_SPACE_FINDING.md
        minimize=False,
        bsz=cfg.bsz,
        k=10,
        verbose=False,
    )

    # seed: an additive, opt-in knob (default None derives the same
    # task_idx-based sequence this function always used) -- mi_orpt/
    # one_step_evaluator.py needs a matched random seed shared across every
    # candidate evaluated against the same background (paper's
    # sec:one-step-pool-evaluation), mirroring peptide's run_bo(seed=...).
    base_seed = seed if seed is not None else task_idx * 10_000

    # cfg.generation_workers > 1: one persistent CPU pool for this task's whole
    # BO loop (mol_generation_pool.py -- real profiling showed a fresh pool per
    # step is a large net loss; see that module's docstring). Created once
    # here, not per step, and torn down when the loop ends regardless of
    # outcome.
    generation_pool = None
    if cfg.generation_workers > 1:
        from mol_generation_pool import create_generation_pool

        generation_pool = create_generation_pool(cfg.generation_workers)
    try:
        for step in range(n_bo_steps):
            lolbo_state.update_surrogate_model()
            mol_acquisition_step(
                lolbo_state, seed_smiles=task.seed_smiles, tau_mol=cfg.tau_mol,
                n_candidates=max(20, cfg.bsz * 4), rng_seed=base_seed + step,
                generation_pool=generation_pool, n_generation_workers=cfg.generation_workers,
            )
            if lolbo_state.tr_state.restart_triggered:
                lolbo_state.initialize_tr_state()
    finally:
        if generation_pool is not None:
            generation_pool.shutdown(wait=True)

    with open(dest_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_x", "train_y"])
        for x, y in zip(lolbo_state.train_x, lolbo_state.train_y.squeeze(-1).tolist()):
            writer.writerow([x, y])

    return dest_csv


def _ensure_mol_constraint_feasible(cfg: MolExperimentConfig, task_idx: int, init_path: Path, scores_path: Path) -> None:
    """Mol analogue of peptide_experiment/steps.py::_ensure_constraint_feasible:
    guarantee a feasible floor (Tanimoto >= cfg.tau_mol against the seed, not
    peptide's edit-distance similarity) via mol_init_candidates.py's own
    guaranteed-feasible generator -- the same one mol_build_mutation_init
    uses -- replacing the least-similar entries, rather than letting LOLBO's
    trust-region logic spin with zero feasible init points."""
    from mol_fingerprint import tanimoto_similarity
    from mol_init_candidates import generate_feasible_candidates_around_seed
    from mol_oracle import EnsembleMolOracle
    from mol_tasks import get_task

    task = get_task(task_idx)
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]

    sims = [tanimoto_similarity(s, task.seed_smiles) for s in seqs]
    n_feasible = sum(1 for s in sims if s is not None and s >= cfg.tau_mol)
    min_feasible = max(1, len(seqs) // 10)
    if n_feasible >= min_feasible:
        return

    n_needed = min_feasible - n_feasible
    print(
        f"[task {task_idx}] only {n_feasible}/{len(seqs)} candidates satisfy tau_mol>={cfg.tau_mol}; "
        f"topping up {n_needed} with guaranteed-feasible mutations of the seed"
    )
    cands, _gen_stats = generate_feasible_candidates_around_seed(
        task.seed_smiles, tau_mol=cfg.tau_mol, n_feasible=n_needed, rng_seed=task_idx
    )
    mutation_smiles = [c for c, _ in cands][:n_needed]
    oracle = EnsembleMolOracle(target_id=task.target_id, target_sequence=task.sequence)
    mutation_scores = oracle.query_oracle(mutation_smiles, input_kind="smiles")

    # Replace the least-similar entries, keeping the most-similar existing ones.
    order = sorted(range(len(seqs)), key=lambda i: (sims[i] if sims[i] is not None else -1.0))
    for i, idx in enumerate(order[: len(mutation_smiles)]):
        seqs[idx] = mutation_smiles[i]
        scores[idx] = mutation_scores[i]

    init_path.write_text("\n".join(seqs) + "\n")
    scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")


def mol_sample_and_build_init(
    cfg: MolExperimentConfig,
    model_path: Path | str,
    task_idx: int,
    work_dir: Path,
    temperature: float | None = None,
    *,
    preloaded_model: tuple | None = None,
) -> tuple[Path, Path]:
    """Sample candidates for mol task `task_idx` from the fine-tuned (or
    base) Qwen checkpoint at `model_path`, score them with the real oracle,
    and return (init_path, scores_path). Mirrors
    peptide_experiment/steps.py::sample_and_build_init's retry-with-widening
    shape: each attempt samples more (pool_multiplier doubling) until
    cfg.init_size feasible+valid candidates are found or max_attempts is
    exhausted, then _pad_to_size / _ensure_mol_constraint_feasible make up
    any shortfall rather than handing LOLBO a starved or empty init set.

    "Feasible+valid" here is stricter than peptide's plain similarity check:
    a raw completion must ALSO parse as a bare SELFIES string
    (mol_prompt.parse_candidate_selfies) and vocab-check-decode to a real
    molecule (mol_sampling.selfies_to_canonical_smiles_vocab_checked) before
    its Tanimoto similarity to the seed is even evaluated -- an under-tuned
    model's malformed/off-grammar output is expected and must not silently
    count as "trying but infeasible."

    temperature: None (default) samples at temperature=1.0 -- every regular
    trajectory-chain call site (trajectory_chain.py::_run_single_task) uses
    this. A float value overrides it, exactly for cfg.mi_candidate_temperature's
    dedicated-pool sampling (mol_experiment/orpt.py::build_orpt_pairs) --
    mirrors peptide's own sample_and_build_init(temperature=...) split
    between "regular deployment-time sampling" and "mi_orpt's own,
    deliberately more exploratory pool sampling." cfg.mi_candidate_temperature
    is deliberately NOT read from cfg directly inside this function -- the
    caller decides, so the two uses can never accidentally collapse into one.

    preloaded_model: None (default) loads model_path fresh, uses it for this
    call only, and frees it before returning -- unchanged behavior for every
    existing call site. A (model, tokenizer, device) triple (from
    mi_orpt/warm_sampling_pool.py's own already-loaded worker state) skips
    both the load and the free -- the model belongs to the calling worker,
    which reuses it across many tasks, not to this one call.
    """
    import torch as _torch

    from mol_fingerprint import tanimoto_similarity
    from mol_oracle import EnsembleMolOracle
    from mol_prompt import parse_candidate_selfies
    from mol_sampling import generate_for_task, load_model_and_tokenizer, selfies_to_canonical_smiles_vocab_checked
    from mol_tasks import get_task

    work_dir.mkdir(parents=True, exist_ok=True)
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    lock_path = work_dir / f"task_{task_idx:04d}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if init_path.exists() and scores_path.exists():
                print(f"[task {task_idx}] init data already exists at {init_path}, skipping")
                # Re-finalize rather than trusting the files outright, same
                # rationale as peptide's own sample_and_build_init: both calls
                # are idempotent/cheap no-ops for a well-formed file.
                _pad_to_size(init_path, scores_path, cfg.init_size)
                _ensure_mol_constraint_feasible(cfg, task_idx, init_path, scores_path)
                return init_path, scores_path

            task = get_task(task_idx)
            if preloaded_model is not None:
                model, tokenizer, device = preloaded_model
            else:
                model, tokenizer, device = load_model_and_tokenizer(model_path)

            effective_temperature = temperature if temperature is not None else 1.0
            pool_multiplier = 1
            max_attempts = 5
            unique_smiles: dict[str, None] = {}
            n_feasible = 0
            for attempt in range(1, max_attempts + 1):
                n_samples = min(cfg.init_size * pool_multiplier, 250)
                raw_completions = generate_for_task(
                    model,
                    tokenizer,
                    task.sequence,
                    task.seed_selfies,
                    device=device,
                    temperature=effective_temperature,
                    top_p=0.95,
                    max_new_tokens=512,
                    num_samples=n_samples,
                )
                for text in raw_completions:
                    parsed = parse_candidate_selfies(text)
                    if parsed is None:
                        continue
                    smiles = selfies_to_canonical_smiles_vocab_checked(parsed)
                    if smiles is None:
                        continue
                    unique_smiles.setdefault(smiles, None)

                sims = {s: tanimoto_similarity(s, task.seed_smiles) for s in unique_smiles}
                n_feasible = sum(1 for sim in sims.values() if sim is not None and sim >= cfg.tau_mol)
                if n_feasible >= cfg.init_size:
                    break
                print(
                    f"[task {task_idx}] only {n_feasible}/{cfg.init_size} feasible "
                    f"({len(unique_smiles)} unique parseable) candidates after attempt "
                    f"{attempt}/{max_attempts}, sampling {min(cfg.init_size * pool_multiplier * 2, 250)} more"
                )
                pool_multiplier *= 2
            else:
                print(
                    f"[task {task_idx}] giving up on reaching {cfg.init_size} feasible candidates "
                    f"after {max_attempts} attempts ({n_feasible} feasible, {len(unique_smiles)} unique "
                    "parseable found); padding/topping up instead"
                )

            if preloaded_model is None:
                # Free the LLM's GPU memory before the caller's next step
                # (mol_run_bo, in the same process) allocates the VAE + GP
                # surrogate on the same device -- see this function's
                # docstring. Skipped when preloaded_model is given: that
                # model belongs to the calling worker, which reuses it
                # across many tasks -- freeing it here would force every
                # subsequent task in that worker to reload from scratch.
                del model
                if device == "cuda":
                    _torch.cuda.empty_cache()

            smiles_list = list(unique_smiles.keys())
            if not smiles_list:
                # The LLM produced zero parseable+valid completions this
                # attempt-cycle (a real, expected outcome for an under-tuned
                # or raw base checkpoint -- see module docstring). The prints
                # above already named this cause; falling back to the seed
                # molecule itself here (trivially Tanimoto=1.0 feasible, a
                # real known-scoreable molecule) gives _pad_to_size /
                # _ensure_mol_constraint_feasible something to build from,
                # rather than crashing on an empty file with no explanation.
                smiles_list = [task.seed_smiles]
            oracle = EnsembleMolOracle(target_id=task.target_id, target_sequence=task.sequence)
            scores = oracle.query_oracle(smiles_list, input_kind="smiles")
            keep = [i for i, s in enumerate(scores) if s == s]  # drop NaN
            smiles_list = [smiles_list[i] for i in keep]
            scores = [scores[i] for i in keep]
            if not smiles_list:
                smiles_list = [task.seed_smiles]
                scores = oracle.query_oracle(smiles_list, input_kind="smiles")

            init_path.write_text("\n".join(smiles_list) + "\n")
            scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")
            _pad_to_size(init_path, scores_path, cfg.init_size)
            _ensure_mol_constraint_feasible(cfg, task_idx, init_path, scores_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return init_path, scores_path

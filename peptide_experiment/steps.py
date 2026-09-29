"""Shared low-level steps for one peptide task: sample+score candidates, and
run one BO trial. Kept separate from trajectory_chain.py because the eval
side (experiments/eval2/) needs the exact same primitives, just pointed at
different output directories.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .config import ExperimentConfig

LOLBO_SCRIPTS_DIR = "optimization/peptides/lolbo_scripts"
FINE_TUNING_DIR = "fine-tuning/peptides"

# sampling_transformers.py generates `samples_per_peptide` sequences in a
# SINGLE model.generate(num_return_sequences=...) call, so memory scales
# with this number directly. Found via a real-scale (init_size=1000) sanity
# check: naively doubling this on every retry (1000 -> 16000) triggered a
# CUDA OOM on attempt 5. Cap it and accumulate multiple smaller calls
# instead of ever-larger single ones.
MAX_SAMPLES_PER_CALL = 2000


def _run(cmd: list, cwd: Path, cfg: ExperimentConfig | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"+ ({cwd}) {printable}", flush=True)
    # When this driver is itself attached to a real terminal (as opposed to a
    # redirected/nohup'd log file), the child inherits that tty on stdin/stdout.
    # Something in the LOLBO dependency chain then thinks it's interactive and
    # opens a pager (man/git/pydoc/rich all pick one up from PAGER/MANPAGER),
    # which blocks forever waiting for a keypress nobody will send -- observed
    # as a 13h+ hung run_lolbo child stuck in do_wait with ~0% CPU. Forcing
    # non-interactive pagers plus a closed stdin closes off every route to
    # that hang, regardless of which dependency triggers it.
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
    """torchtune==0.4.0 (pinned in this container -- /opt/bolt-constraints.txt,
    a torch/torchvision/torchao/transformers-CUDA-matched stack) saves each
    epoch's full merged (base+LoRA) checkpoint as separate
    hf_model_<cpt_idx>_<epoch>.pt shards directly in output_dir -- raw
    torch.save with a non-standard filename, no model.safetensors.index.json.
    Three downstream consumers all need a real HF-standard checkpoint dir
    here instead of that:
      - fine-tuning/peptides/sampling_transformers.py's
        AutoModelForCausalLM.from_pretrained()/AutoTokenizer.from_pretrained()
        (task sampling)
      - torchtune's own FullModelHFCheckpointer.load_checkpoint(), reused
        verbatim (same hardcoded checkpoint_files list from the
        torchtune_config YAML) to reload this milestone's checkpoint as the
        DPO stage's base model (peptide_experiment/orpt.py)
      - mi_orpt/likelihood.py's own torchtune-config-instantiated
        checkpointer/tokenizer, for reference-model log-likelihood scoring

    This converts each hf_model_<cpt_idx>_<epoch>.pt shard into
    model-<cpt_idx>-of-<N>.safetensors -- the exact filenames every
    torchtune_config's checkpoint_files list hardcodes -- then copies the
    base checkpoint's OWN model.safetensors.index.json verbatim: LoRA
    merging only changes tensor values, never which keys exist or which
    shard they came from (confirmed: hf_model_<cpt_idx>_<epoch>.pt's cpt_idx
    numbering is built by torchtune from loading those same base
    checkpoint_files in order), so the original weight_map/total_size stay
    exactly correct. Also copies the base checkpoint's tokenizer files
    alongside -- every tune-run invocation in this repo only ever points
    tokenizer.path/merges_file at the BASE checkpoint, so no per-milestone
    copy is ever produced automatically, yet all three consumers above
    default to loading the tokenizer from the checkpoint dir itself.

    Idempotent: returns immediately once model.safetensors.index.json
    already exists in ckpt_dir. Shared by both the SFT
    (trajectory_chain.py) and ORPT (orpt.py) training steps.

    Our environment (previously an A100 box, now pinned to a different
    torchtune version than the 0.4.0 above) runs torchtune>=0.6, whose
    FullModelHFCheckpointer already writes a full HF-standard checkpoint
    (safetensors shards + model.safetensors.index.json + tokenizer files)
    unprompted -- just nested under ckpt_dir/epoch_<epoch>/ instead of
    directly in ckpt_dir. In that case there's nothing to convert; just
    flatten epoch_<epoch>/ up into ckpt_dir.
    """
    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        return ckpt_dir

    epoch_dir = ckpt_dir / f"epoch_{epoch}"
    if (epoch_dir / "model.safetensors.index.json").exists():
        # list(...) up front: shutil.move below removes each item from
        # epoch_dir as it goes, and mutating a directory while iterating
        # os.scandir over it (what Path.iterdir() uses) is unspecified.
        #
        # A checkpointer using ckpt_dir as its OWN checkpointer.checkpoint_dir
        # (e.g. ORPT-<m>'s DPO stage loading pi_ref from BOLT-<m>) copies
        # "every file in ckpt_dir" into its own new epoch_<e>/ output as a
        # side effect of loading it (torchtune's own save_checkpoint) -- if
        # ckpt_dir's SOURCE still had stray epoch_<n>/recipe_state/ clutter
        # (cleanup_intermediate_epochs should prevent this, but don't rely
        # on ordering across two different training stages), those get
        # copied in here too, nested one level deeper, and a copied-in item
        # literally named the same as this very epoch_dir (e.g. "epoch_0"
        # inside epoch_dir itself) makes shutil.move's "move into an
        # existing directory" fallback collide with epoch_dir's own name.
        # Discard such clutter outright instead of moving it -- it was never
        # part of the real HF checkpoint bundle.
        for item in list(epoch_dir.iterdir()):
            is_stray_epoch_dir = item.is_dir() and item.name.startswith("epoch_") and item.name[len("epoch_") :].isdigit()
            if item.is_dir() and (item.name == "recipe_state" or is_stray_epoch_dir):
                shutil.rmtree(item)
                continue
            shutil.move(str(item), str(ckpt_dir / item.name))
        epoch_dir.rmdir()
        return ckpt_dir

    import torch
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
        # safetensors refuses to write two keys that alias the same storage
        # (e.g. tied embeddings) -- shouldn't occur here (torchtune's own
        # weight_map, built from loading the base checkpoint's un-tied
        # 434-key export, already excludes the tied lm_head.weight key), but
        # cloning any duplicate is a cheap, unconditional guard rather than
        # a discovery made the hard way mid-run.
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
    """torchtune's lora_finetune_distributed/lora_dpo_distributed recipes save
    a full merged checkpoint after every epoch with no "final epoch only"
    option, so training for N epochs leaves N full multi-GB checkpoints on
    disk. With torchtune==0.4.0's flat (non-nested) output_dir layout (see
    materialize_hf_checkpoint's docstring), those are hf_model_<cpt_idx>_<e>.pt
    / adapter_<e>.pt files distinguished only by the epoch number embedded in
    each filename, not by an epoch_<e>/ subdirectory as an older torchtune
    version would have produced. Deletes every non-final epoch's raw shard/
    adapter files, and -- once materialize_hf_checkpoint has converted them --
    final_epoch's own now-redundant raw .pt shards too. Also drops the
    intermediate-checkpoint-only recipe_state.pt (irrelevant once training
    has finished). Shared by both the SFT (trajectory_chain.py) and ORPT
    (orpt.py) training steps.

    torchtune>=0.6 path (2026-09 diagnostic session, real milestone-600 run):
    the glob patterns above only ever match torchtune==0.4.0's flat layout
    and silently find nothing under torchtune>=0.6, which instead nests each
    epoch's full HF-format checkpoint under ckpt_dir/epoch_<e>/ (see
    materialize_hf_checkpoint's own torchtune>=0.6 branch, which flattens
    only epoch_<final_epoch>/ up to ckpt_dir and leaves the other epoch_<e>/
    dirs + epoch_<final_epoch>/'s own recipe_state/ untouched). Left alone,
    those stray directories are more than wasted disk: a downstream
    checkpointer using ckpt_dir as its OWN checkpointer.checkpoint_dir (e.g.
    ORPT-<m>'s DPO stage loading pi_ref from BOLT-<m>) copies "everything in
    ckpt_dir" into its own new epoch_<e>/ output as a side effect of loading
    it, silently duplicating this same clutter one level deeper -- and if a
    copied-in item happens to collide with that new epoch_<e>/ directory's
    own name, materialize_hf_checkpoint's flatten can hit a real
    shutil.Error (observed on a live milestone-10 ORPT run). Delete every
    epoch_<n> subdirectory (n != final_epoch, or n == final_epoch once
    materialized) and any recipe_state/ directory the same way the
    torchtune==0.4.0 branch above deletes its own flat-file equivalents.
    """
    materialized = (ckpt_dir / "model.safetensors.index.json").exists()
    for pattern in ("hf_model_*_*.pt", "adapter_*.pt"):
        for f in ckpt_dir.glob(pattern):
            try:
                epoch = int(f.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue  # defensive -- every current match of these globs is epoch-suffixed
            if epoch != final_epoch or materialized:
                f.unlink()
    recipe_state = ckpt_dir / "recipe_state.pt"
    if recipe_state.exists():
        recipe_state.unlink()
    for d in ckpt_dir.glob("epoch_*"):
        if not d.is_dir():
            continue
        suffix = d.name[len("epoch_") :]
        if not suffix.isdigit():
            continue
        epoch = int(suffix)
        if epoch != final_epoch or materialized:
            shutil.rmtree(d)
    recipe_state_dir = ckpt_dir / "recipe_state"
    if recipe_state_dir.is_dir():
        shutil.rmtree(recipe_state_dir)


def distributed_finetune_launch(
    cfg: ExperimentConfig, torchtune_config_name: str, fine_tuning_dir: Path
) -> tuple[list[str], list[str], ExperimentConfig]:
    """Returns (torchrun-level flags to insert before `--config`, extra
    `tune run` config overrides to append after it, a possibly-GPU-widened
    cfg to pass as _run()'s `cfg=` for CUDA_VISIBLE_DEVICES) for SFT/DPO
    fine-tuning. Shared by trajectory_chain.py::train_milestone() and
    orpt.py::train_orpt_milestone().

    The torchrun-level flags carry --master-port: `tune run`'s distributed
    launcher defaults to a fixed rendezvous port (29500) regardless of
    which GPUs are used, so two concurrent multi-GPU launches on the same
    host collide on it even with fully disjoint mi_parallel_gpus (confirmed
    the hard way this session -- one pipeline's `tune run` failed with
    EADDRINUSE while another's, on different GPUs entirely, was already
    bound to it). Deriving the port from mi_parallel_gpus[0] (29500 + that
    GPU id) gives each disjoint GPU group its own fixed, deterministic port
    with no new config field -- e.g. GPUs [0..3] -> 29500, [4..7] -> 29504.

    cfg.mi_parallel_gpus, if set, is reused here too -- not just for
    matched-intervention pair construction's warm-worker pool (see
    orpt.py::build_orpt_pairs) -- per explicit user request: the same
    reserved GPU pool doubles as the fine-tuning device list, since this
    host has enough headroom for both uses. Unset (default): completely
    unchanged behavior, --nproc_per_node stays 1, cfg passed through as-is.

    The torchtune config YAML's own `batch_size` is a PER-DEVICE value,
    passed through to every rank unchanged (2026-09: previously divided by
    nproc_per_node to hold the *global* effective batch constant across a
    1-GPU vs. N-GPU launch of the same YAML -- deliberately dropped per
    explicit user request to instead max out each A100's 40GB: FSDP's
    per-rank data-parallel batch is what determines a single GPU's memory
    footprint, so a YAML value tuned to that ceiling (empirically ~32 for
    this repo's 3B-parameter/LoRA-rank-16 SFT config: batch=32 measured at
    ~27GiB peak, batch=64 OOMs, on a single 40GB A100 -- see
    imp_plan/ or this session's memory probe for the raw numbers) now scales
    the *global* effective batch linearly with GPU count instead of holding
    it fixed -- standard data-parallel practice, and the entire reason
    mi_parallel_gpus exists here (wall-clock AND throughput scaling, not
    just wall-clock at a fixed global batch). Read directly from the
    torchtune config YAML since ExperimentConfig itself has no batch_size
    field for SFT/DPO (that lives entirely in the recipe config, unlike
    cfg.bsz which is the BO acquisition batch size).
    """
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


def _similarity(seq: str, reference_seq: str) -> float:
    from Levenshtein import distance as edit_distance

    length = len(reference_seq)
    return (length - edit_distance(seq, reference_seq)) / length


def _pad_to_size(init_path: Path, scores_path: Path, target_size: int) -> None:
    """Known failure mode (from prior smoke testing): an untuned/low-diversity
    model can plateau well below target_size unique candidates no matter how
    many retries. Pad by resampling with replacement rather than crashing.
    """
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


def _ensure_constraint_feasible(
    cfg: ExperimentConfig, task_idx: int, init_path: Path, scores_path: Path
) -> None:
    """Found via smoke testing: an under-trained (or raw base) model can
    produce candidates that are ALL infeasible under the similarity
    constraint (none within cfg.similarity_threshold of the reference
    peptide). With zero feasible init points, LOLBO's trust-region logic
    spins in a near-infinite loop (rapid repeated 0% progress bars, no
    oracle-call progress) instead of erroring out. Guarantee a feasible
    floor by topping up with mutations of the reference sequence itself
    (the same method stbo_optimization.py uses), replacing the
    least-similar entries.
    """
    from apex_oracle import apex_wrapper
    from apex_oracle.init_data.create_mutations import generate_unique_mutations
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    reference_seq = REFERENCE_SEQUENCE[task_idx]
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]

    sims = [_similarity(s, reference_seq) for s in seqs]
    n_feasible = sum(1 for s in sims if s >= cfg.similarity_threshold)
    min_feasible = max(1, len(seqs) // 10)
    if n_feasible >= min_feasible:
        return

    n_needed = min_feasible - n_feasible
    print(
        f"[task {task_idx}] only {n_feasible}/{len(seqs)} candidates satisfy the "
        f"similarity>={cfg.similarity_threshold} constraint; topping up {n_needed} "
        "with guaranteed-feasible mutations of the reference sequence"
    )
    mutations = generate_unique_mutations(
        [reference_seq],
        num_mutations=n_needed,
        max_mutation_distance=1.0 - cfg.similarity_threshold,
    )[0]
    mutation_scores = list(-apex_wrapper(mutations)[:, 0])

    # Replace the least-similar entries, keeping the most-similar existing ones.
    order = sorted(range(len(seqs)), key=lambda i: sims[i])
    for i, idx in enumerate(order[:n_needed]):
        seqs[idx] = mutations[i]
        scores[idx] = mutation_scores[i]

    init_path.write_text("\n".join(seqs) + "\n")
    scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")


def build_mutation_init(
    cfg: ExperimentConfig,
    task_idx: int,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Build init candidates for a task with no fine-tuned checkpoint yet
    (i.e. before the first milestone) via mutations of the reference
    sequence, scored by the oracle -- the same method stbo_optimization.py
    uses, and the same method the original codebase used to precompute
    apex_oracle/init_data/seed_0_init.txt (see create_mutations.py).
    Sampling these tasks from the raw, untuned base model instead produces
    degenerate/low-diversity output (confirmed: see imp_plan progress log).
    """
    from apex_oracle import apex_wrapper
    from apex_oracle.init_data.create_mutations import generate_unique_mutations
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    work_dir.mkdir(parents=True, exist_ok=True)
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    lock_path = work_dir / f"task_{task_idx:04d}.lock"

    # generate_unique_mutations is unseeded, so when this is the SHARED
    # canonical pool (fixed_target_bo.py's SHARED_INIT_ARMS -- several GPU
    # worker processes all calling this for the exact same (work_dir,
    # task_idx) at sweep startup) the old exists-check-then-write here was a
    # real race: multiple processes could all see "not there yet" before
    # any of them finished writing, each generate their OWN different
    # mutations, and each copy their own version out -- confirmed in
    # practice (three arms silently ended up on three different pools from
    # one concurrent launch). The flock makes the whole check+generate+write
    # one atomic section per task_idx, so exactly one process ever
    # generates and every other one correctly waits and then reuses it.
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if init_path.exists() and scores_path.exists():
                print(f"[task {task_idx}] init data already exists at {init_path}, skipping")
                return init_path, scores_path

            reference_seq = REFERENCE_SEQUENCE[task_idx]
            mutations = generate_unique_mutations(
                [reference_seq],
                num_mutations=cfg.init_size,
                max_mutation_distance=1.0 - cfg.similarity_threshold,
            )[0]
            scores = list(-apex_wrapper(mutations)[:, 0])

            init_path.write_text("\n".join(mutations) + "\n")
            scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")
            _pad_to_size(init_path, scores_path, cfg.init_size)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return init_path, scores_path


def sample_and_build_init(
    cfg: ExperimentConfig,
    model_path: Path | str,
    task_idx: int,
    work_dir: Path,
    temperature: float | None = None,
    temperature_step: float = 0.0,
    *,
    sampling_pool: "object | None" = None,
) -> tuple[Path, Path]:
    """Sample candidates for peptide task `task_idx` from `model_path`
    (a checkpoint dir, or the raw base model for task 0 / pre-milestone
    tasks), score them with the APEX oracle, and return (init_path,
    scores_path) ready to hand to the BO entry point.

    temperature: None (default) leaves sampling_transformers.py at its own
    default (1) -- every existing call site's behavior is unchanged. A float
    value overrides it, e.g. for mi_orpt's dedicated candidate-pool sampling
    (cfg.mi_candidate_temperature) without affecting any other caller.

    temperature_step: added to `temperature` on each successive retry attempt
    below (attempt 1 uses `temperature` itself, attempt 2 uses
    `temperature + temperature_step`, ...). No effect when `temperature` is
    None. Default 0.0 (flat, unchanged behavior) -- eval2/compute/
    generate_raw.py's cfg.eval_raw_temperature_step is the one caller that
    sets this non-zero, to widen sampling entropy on top of the existing
    pool_multiplier widening when a plateaued unique-candidate count suggests
    entropy, not sample count, is the retry bottleneck.

    sampling_pool: None (default) keeps the original behavior exactly -- one
    fresh `sampling_transformers.py` subprocess per attempt. A
    mi_orpt.warm_sampling_pool.SamplingPool for THIS call's GPU instead reuses
    a worker that already holds `model_path` resident, removing the ~31s
    per-call checkpoint load that dominated this step (~38s/call for ~7s of
    generation). Typed loosely so this module needs no import of the pool
    module, which must stay free of eager torch imports. The caller is
    responsible for the pool matching `model_path` -- the worker re-checks and
    fails loudly if it doesn't. Falls back to the subprocess path if the pool
    breaks, so a dead worker degrades throughput rather than the run.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    if init_path.exists() and scores_path.exists():
        print(f"[task {task_idx}] init data already exists at {init_path}, skipping")
        # Re-finalize rather than trusting the files outright: both are written
        # INSIDE the retry loop below, after every attempt, so a run killed
        # mid-retry leaves a file that exists but never reached _pad_to_size /
        # _ensure_constraint_feasible -- i.e. it can be short of cfg.init_size
        # and below the feasibility floor, yet still look "already done" here.
        # That failure is silent and only surfaces much later as an
        # unexplainably undersized eligible bank in mi_orpt pair construction
        # (hit for real: two tasks sat at 98/110 feasible against the 115 the
        # bank needed, purely because their sampling was interrupted). Both
        # calls are idempotent and cheap no-ops for a well-formed file --
        # _ensure_constraint_feasible returns before touching the oracle when
        # the floor is already met -- so this costs nothing in the normal case.
        _pad_to_size(init_path, scores_path, cfg.init_size)
        _ensure_constraint_feasible(cfg, task_idx, init_path, scores_path)
        return init_path, scores_path

    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    reference_seq = REFERENCE_SEQUENCE[task_idx]
    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR

    pool_multiplier = 1
    max_attempts = 5
    n_unique = 0
    n_feasible = 0
    sample_jsonls: list[Path] = []
    for attempt in range(1, max_attempts + 1):
        samples_per_peptide = min(cfg.init_size * pool_multiplier, MAX_SAMPLES_PER_CALL)
        attempt_temperature = None
        if temperature is not None:
            attempt_temperature = temperature + (attempt - 1) * temperature_step
        attempt_jsonl = work_dir / f"task_{task_idx:04d}_sampled_attempt{attempt}.jsonl"
        cmd = [
            sys.executable,
            "sampling_transformers.py",
            "--model-path",
            model_path,
            "--start-index",
            task_idx,
            "--num-peptides",
            1,
            "--samples-per-peptide",
            samples_per_peptide,
            "--output-file",
            attempt_jsonl,
        ]
        if attempt_temperature is not None:
            cmd += ["--temperature", attempt_temperature]

        used_pool = False
        if sampling_pool is not None and not sampling_pool.broken:
            from concurrent.futures.process import BrokenProcessPool

            try:
                # Mirror _run()'s own "+ (cwd) cmd" echo, including the literal
                # `sampling_transformers` token and the same flag names: the
                # per-task sampling timings in this repo are read off these log
                # lines, and the pooled path would otherwise go dark.
                print(
                    f"+ [warm-sampling-pool gpu={sampling_pool.gpu}] sampling_transformers "
                    f"--model-path {model_path} --start-index {task_idx} "
                    f"--samples-per-peptide {samples_per_peptide} --output-file {attempt_jsonl}"
                    + (f" --temperature {attempt_temperature}" if attempt_temperature is not None else "")
                )
                sampling_pool.generate(
                    task_idx=task_idx,
                    samples_per_peptide=samples_per_peptide,
                    output_file=attempt_jsonl,
                    temperature=attempt_temperature,
                )
                used_pool = True
            except BrokenProcessPool as e:
                # Latched on the pool itself, so later attempts/tasks on this
                # GPU go straight to the CLI instead of re-raising every time.
                print(
                    f"[task {task_idx}] warm sampling pool on gpu {sampling_pool.gpu} is broken "
                    f"({e!r}); falling back to the subprocess path for the rest of this segment"
                )
        if not used_pool:
            _run(cmd, cwd=fine_tuning_dir, cfg=cfg)
        sample_jsonls.append(attempt_jsonl)
        _run(
            [
                sys.executable,
                "make_initialization_data.py",
                "--input-jsonl",
                *sample_jsonls,
                "--output-init",
                init_path,
                "--output-scores",
                scores_path,
                "--deduplicate",
            ],
            cwd=fine_tuning_dir / "sampled_output_from_ft",
            cfg=cfg,
        )
        seqs = [line.strip() for line in init_path.read_text().splitlines() if line.strip()]
        n_unique = len(seqs)
        # Stop on FEASIBLE count, not raw unique count: a task can reach
        # cfg.init_size unique sequences in one attempt while few of them are
        # within cfg.similarity_threshold of the reference (observed for real
        # on eval2/arm_specs/fig1b.yaml's ORPT-H1 arm -- one task hit 1000+
        # unique candidates on attempt 1 yet only 54 were feasible), which
        # used to end the loop before temperature/pool_multiplier ever got a
        # chance to widen the search. feasible_pool_with_draw_counts
        # (eval2/core/pools.py) re-derives its own feasible-only pool from
        # the raw generations regardless of what this loop decides, so this
        # only affects how hard we try -- never what's fed to a strict
        # downstream feasibility filter.
        n_feasible = sum(1 for s in seqs if _similarity(s, reference_seq) >= cfg.similarity_threshold)
        if n_feasible >= cfg.init_size:
            break
        print(
            f"[task {task_idx}] only {n_feasible}/{cfg.init_size} feasible ({n_unique} unique) "
            f"candidates after dedup across {len(sample_jsonls)} sampling call(s) "
            f"(attempt {attempt}/{max_attempts}), sampling {samples_per_peptide} more"
        )
        pool_multiplier *= 2
    else:
        print(
            f"[task {task_idx}] giving up on reaching {cfg.init_size} feasible candidates "
            f"after {max_attempts} attempts ({n_feasible} feasible, {n_unique} unique found); padding instead"
        )

    _pad_to_size(init_path, scores_path, cfg.init_size)
    _ensure_constraint_feasible(cfg, task_idx, init_path, scores_path)
    return init_path, scores_path


def run_bo(
    cfg: ExperimentConfig,
    task_idx: int,
    work_dir: Path,
    run_id: str,
    init_path: Path | None = None,
    scores_path: Path | None = None,
    stbo: bool = False,
    seed: int | None = None,
    pretrained_surrogate_path: Path | None = None,
    surrogate_type: str | None = None,
    poe_manifest_path: Path | None = None,
    update_e2e: bool | None = None,
) -> Path:
    """Run one single-task BO trial (BOLT arm if init_path/scores_path are
    given, STBO arm if stbo=True) and return the path to its collected-data
    CSV (train_x, train_y), copied into `work_dir` for permanence.

    seed is an additive, opt-in knob (default None reproduces the exact
    subprocess CLI this function has always built): peptide_experiment/
    mi_orpt/one_step_evaluator.py::run_candidates_one_step needs a matched
    random seed shared across every candidate evaluated against the same
    background (paper/method.tex sec:one-step-pool-evaluation).
    Forwards directly to Optimize's own (already-existing, otherwise-unused)
    `--seed` constructor kwarg. VAE pretrained-checkpoint loading is
    controlled by cfg.use_pretrained_vae (see config.py).

    pretrained_surrogate_path: None (default, unchanged): LOLBOState fits a
    fresh GP surrogate from scratch, same as ever. A path (shared-surrogate
    MTBO baseline, mtbo.py::train_mtbo_surrogate's output): forwarded to
    Optimize's pretrained_surrogate_path constructor kwarg, along with the
    exact inducing-point count recorded in that checkpoint's sidecar
    surrogate_meta.json (lolbo.py::LOLBOState needs the count to construct a
    matching-shape model before load_state_dict).

    surrogate_type/poe_manifest_path/update_e2e: None (default, unchanged):
    all three omitted from the CLI, Optimize's own defaults used ("gp_dkl",
    no manifest, update_e2e=True) -- every existing call site unaffected.
    GP-expert-transfer baseline (gp_expert_transfer.py): pass
    surrogate_type="gp_poe" + poe_manifest_path=<a {"experts": [...]} JSON
    manifest> + update_e2e=False (required whenever surrogate_type="gp_poe"
    -- a frozen K-expert ensemble has no single model to fine-tune
    end-to-end; see lolbo.py::LOLBOState.update_models_e2e's guard).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = work_dir / f"task_{task_idx:04d}.csv"
    if dest_csv.exists():
        print(f"[{run_id} task {task_idx}] trajectory already exists at {dest_csv}, skipping")
        return dest_csv

    lolbo_scripts_dir = cfg.bolt_root / LOLBO_SCRIPTS_DIR
    run_name = f"{run_id}_task_{task_idx:04d}"
    script = "stbo_optimization.py" if stbo else "info_transformer_vae_optimization.py"

    cmd = [
        sys.executable,
        script,
        "--task_id",
        "apex",
        "--max_n_oracle_calls",
        cfg.oracle_budget,
        "--bsz",
        cfg.bsz,
        "--constraint_function_ids",
        "[similarity]",
        "--constraint_thresholds",
        f"[{cfg.similarity_threshold}]",
        "--constraint_types",
        f"[{task_idx}]",
        "--num_initialization_points",
        cfg.init_size,
        "--init_n_update_epochs",
        20,
        "--max_string_length",
        30,
        "--task_specific_args",
        f"[{cfg.task_specific_args}]",
        "--init_offset_helper",
        task_idx,
        "--track_with_wandb",
        False,
        "--wandb_project_name",
        cfg.experiment_id,
        "--wandb_run_name",
        run_name,
    ]
    if init_path is not None:
        # Previously gated on `not stbo` -- STBO's own load_train_data() now
        # accepts an injected pool too (fixed_target_bo.py's
        # SHARED_INIT_ARMS), so this only needs to check whether the caller
        # actually has one to inject, same as every other arm.
        cmd += [
            "--init_data_path",
            init_path,
            "--init_scores_path",
            scores_path,
        ]
    if seed is not None:
        cmd += ["--seed", seed]
    if not cfg.use_pretrained_vae:
        cmd += ["--path_to_vae_statedict", ""]
    if pretrained_surrogate_path is not None:
        meta_path = Path(pretrained_surrogate_path).with_name("surrogate_meta.json")
        meta = json.loads(meta_path.read_text())
        cmd += [
            "--pretrained_surrogate_path",
            str(pretrained_surrogate_path),
            "--pretrained_surrogate_num_inducing",
            meta["num_inducing_points"],
        ]
    if surrogate_type is not None:
        cmd += ["--surrogate_type", surrogate_type]
    if poe_manifest_path is not None:
        cmd += ["--poe_manifest_path", str(poe_manifest_path)]
    if update_e2e is not None:
        cmd += ["--update_e2e", update_e2e]
    cmd.append("run_lolbo")

    _run(cmd, cwd=lolbo_scripts_dir, cfg=cfg)

    produced_csv = (
        lolbo_scripts_dir
        / "optimization_all_collected_data"
        / f"{cfg.experiment_id}_{run_name}_all-data-collected.csv"
    )
    if not produced_csv.exists():
        raise RuntimeError(f"Expected BO output at {produced_csv}, but it wasn't produced")
    shutil.copy(produced_csv, dest_csv)
    return dest_csv

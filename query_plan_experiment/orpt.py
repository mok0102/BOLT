from .timing import timed_operation
from query_plan_timing import event, span
"""Build MI preferences and train DPO on the same milestone's merged SFT model."""
import gc
import json
import random
from pathlib import Path
import yaml
from .steps import _run, FINE_TUNING_DIR, WORKLOAD_DIR, cleanup_intermediate_epochs

def check_training_size(data, recipe, epochs):
    n = sum(bool(line.strip()) for line in Path(data).read_text().splitlines())
    settings = yaml.safe_load(Path(recipe).read_text())
    batch = settings["batch_size"]
    steps = (n // batch) // settings.get("gradient_accumulation_steps", 1)
    if epochs < 1 or steps < 1:
        raise RuntimeError(f"No optimizer steps: {n} rows, batch={batch}, epochs={epochs}. Reduce batch size.")
    print(f"Training: {n} rows, {steps} steps/epoch, {steps*epochs} total steps")

def sql_for_workload(cfg, workload):
    import re
    group, name = workload.lower().split("_", 1)
    base = cfg.bolt_root / WORKLOAD_DIR
    if group == "ceb":
        folder = re.match(r"\d+[a-z]+", name).group()
        path = base / "ceb-3k" / folder / f"{name}.sql"
    elif group == "job":
        path = base / "job" / f"{name}.sql"
    else:
        raise ValueError(workload)
    return path.read_text()

@timed_operation("mi_pair_construction")
def build_orpt_pairs(cfg, milestone):
    import torch
    from .mi_orpt.candidate_bank import build_eligible_bank, min_bank_size_needed
    from .mi_orpt.likelihood import load_reference, score_sequences, SYSTEM_PROMPT
    from .mi_orpt.pair_construction import construct_pairs_for_task
    target = cfg.orpt_pairs_dir / f"orpt_pairs_{milestone}.jsonl"
    if target.exists() and target.stat().st_size:
        event("cache_hit", 0, artifact=str(target))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for workload in cfg.train_task_workloads()[:milestone]:
        bank = build_eligible_bank(cfg.trajectories_csv_dir / f"{workload}.csv")
        minimum = min_bank_size_needed(cfg.init_size, cfg.mi_max_candidates_per_task)
        print(f"[MI {workload}] {len(bank)} unique completed plans; ideal {minimum}, minimum 3 with background padding")
        if len(bank) >= 3:
            tasks.append((workload, bank, sql_for_workload(cfg, workload)))
    if not tasks:
        raise RuntimeError("MI requires at least 3 distinct completed plans in one task (2 comparison candidates + 1 background); duplicates cannot create distinct candidates")
    with span("mi_reference_model_load"):
        model, tokenizer = load_reference(cfg.milestone_checkpoint_dir(milestone), cfg.base_checkpoint_dir)
    scored = []
    for w, b, sql in tasks:
        with span("mi_reference_likelihood", workload=w, candidates=len(b)):
            scores = score_sequences(model, tokenizer, sql, [c.seq for c in b])
        scored.append((w,b,sql,scores))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    records = []
    for i,(workload,bank,sql,scores) in enumerate(scored):
        records.extend(construct_pairs_for_task(
            cfg, workload, bank, sql, scores, cfg.init_size, cfg.mi_target_pairs_per_task,
            cfg.mi_max_candidates_per_task, cfg.mi_num_backgrounds, cfg.mi_tau_q,
            cfg.mi_z_min, cfg.mi_delta_t, cfg.mi_bo_steps,
            cfg.run_dir / "mi_evaluations" / f"milestone_{milestone}" / workload,
            random.Random(cfg.mi_seed+i)))
    (target.with_suffix(".diagnostics.json")).write_text(json.dumps(records,indent=2))
    if not records:
        raise RuntimeError("No MI preference pairs passed the reliability filter; no DPO checkpoint created")
    rows = []
    for r in records:
        prefix = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":r["reference_sequence"]}]
        rows.append({arm: prefix+[{"role":"assistant","content":r[f"{arm}_sequence"]}] for arm in ["chosen","rejected"]})
    tmp = target.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row)+"\n" for row in rows))
    tmp.replace(target)
    return target

@timed_operation("dpo")
def train_orpt_milestone(cfg, milestone):
    final = cfg.orpt_checkpoint_dir(milestone)
    if final.exists():
        event("cache_hit", 0, artifact=str(final))
        return final
    bolt = cfg.milestone_checkpoint_dir(milestone)
    if not bolt.exists():
        raise RuntimeError(f"Missing SFT checkpoint: {bolt}")
    pairs = build_orpt_pairs(cfg, milestone)
    directory = cfg.bolt_root / FINE_TUNING_DIR
    check_training_size(pairs, directory / "torchtune_config" / cfg.orpt_torchtune_config, cfg.orpt_epochs)
    out = cfg.checkpoints_dir / f"ORPT-{milestone}"
    _run(["tune","run","--nnodes","1","--nproc_per_node","1",cfg.orpt_torchtune_recipe,
          "--config",f"torchtune_config/{cfg.orpt_torchtune_config}",f"output_dir={out}",
          f"dataset.data_files={pairs}",f"checkpointer.checkpoint_dir={bolt}",
          f"tokenizer.path={cfg.base_checkpoint_dir}/vocab.json",
          f"tokenizer.merges_file={cfg.base_checkpoint_dir}/merges.txt",
          f"epochs={cfg.orpt_epochs}",f"loss.beta={cfg.orpt_beta}",f"optimizer.lr={cfg.orpt_lr}",
          f"metric_logger.log_dir={cfg.tensorboard_dir / f'ORPT-{milestone}'}","seed=42"],cwd=directory,cfg=cfg)
    if not final.exists():
        raise RuntimeError(f"DPO did not produce {final}")
    cleanup_intermediate_epochs(out,final)
    return final

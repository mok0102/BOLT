"""Build a *fixed-size* init pool via real rejection sampling (no synthetic
top-up), for each of several target pool sizes, then run real BO on it --
comparing models on equal footing at a chosen initialization-pool size.

This is the compute engine behind fig:main-bo (paper/experiments.tex
sec:main-results, "Full-budget optimization") and, at a fixed bo_calls /
target_pool_size slice, fig:scaling.

If a task's already-generated raw proposals don't contain `target` feasible
unique candidates, that task is skipped for that target (no on-demand extra
sampling) and counted in a coverage_rate metric. Coverage is reported
separately from BO performance on purpose: an arm that only clears a given
target for a few tasks is a real finding (a naive-DPO arm paying for its own
constraint violations in reduced coverage), not something to average away.

STBO/MTBO/OptFormer/POGPE/SGPE/LLAMBO are self-seeding baselines: they never
go through generate_raw.py, and build their own target-sized init pool
directly here.

Each covered (task, target) launches one real BO run -- start with
--limit-tasks and a single target size before attempting a full sweep.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

from ..core.arms import ArmSpec, bo_work_dir_for, raw_dir_for
from ..core.pools import bo_k_checkpoints, build_bo_pool, write_csv

SELF_SEEDING_ARMS = ("STBO", "MTBO", "OptFormer", "POGPE", "SGPE", "LLAMBO")

DEFAULT_TARGET_POOL_SIZES = [10, 20, 50]

COVERAGE_FIELDS = [
    "arm", "milestone", "task_set", "target_pool_size",
    "n_tasks", "n_ran_bo", "coverage_rate", "mean_best_feasible_incumbent",
    "mean_rejection_rate",
]
PER_TASK_FIELDS = [
    "arm", "milestone", "task_set", "task_idx", "target_pool_size",
    "draws_used", "rejection_rate", "best_feasible_incumbent", "bo_calls", "best_mic",
    "llambo_terminated_early", "llambo_input_tokens_used",
]
SUMMARY_FIELDS = ["arm", "milestone", "task_set", "target_pool_size", "bo_calls", "n_tasks_ran_bo", "mean_best_mic"]

# MTBO/POGPE/SGPE/OptFormer/STBO all self-seed via steps.build_mutation_init
# (or, for STBO, stbo_optimization.py's now-optional injected-pool path)
# with no fixed RNG seed -- left alone, each arm draws a DIFFERENT random
# mutation pool for the same task even though the generation method is
# identical, confounding "is this baseline's own search better" with "did
# it luck into a better starting point". SHARED_INIT_ARMS get ONE canonical
# pool per (task_set, task_idx, target) instead, generated once and reused
# by all five -- see _seed_shared_mutation_init. LLAMBO is deliberately
# excluded: it isn't part of this comparison right now.
SHARED_INIT_ARMS = ("MTBO", "OptFormer", "POGPE", "SGPE", "STBO")


def _seed_shared_mutation_init(cfg, spec: ArmSpec, task_set: str, task_idx: int, target: int, work_dir: Path) -> None:
    """Generate (or reuse) the one canonical mutation-init pool for this
    (task_set, task_idx, target), then copy it into this arm's own work_dir
    under the exact filenames build_mutation_init expects there. Its own
    exists() check then skips regeneration -- so no caller needs to change,
    including run_optformer_bo, which calls build_mutation_init internally
    with work_dir as the directory and never sees this indirection."""
    from peptide_experiment.steps import build_mutation_init

    shared_dir = spec.run_dir / "eval_fixed_target_bo" / task_set / f"_shared_mutation_init__target{target}"
    shared_init, shared_scores = build_mutation_init(cfg, task_idx, shared_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    dest_init = work_dir / f"task_{task_idx:04d}_init.txt"
    dest_scores = work_dir / f"task_{task_idx:04d}_scores.csv"
    if not dest_init.exists():
        dest_init.write_text(shared_init.read_text())
    if not dest_scores.exists():
        dest_scores.write_text(shared_scores.read_text())


def run_for_spec_task_set_target(
    dom, cfg, spec: ArmSpec, task_set: str, target: int, task_ids: list,
) -> tuple[dict, list[dict]]:
    raw_dir = raw_dir_for(spec, task_set)
    work_dir = bo_work_dir_for(spec, task_set, target)
    run_id = f"fixedtarget-{task_set}-{spec.arm}-{spec.milestone}-t{target}"

    n_ran_bo = 0
    rejection_rates: list[float] = []
    feasible_incumbents: list[float] = []
    bo_rows: list[dict] = []

    # OptFormer/LLAMBO call run_optformer_bo/run_llambo_bo in-process (no
    # subprocess boundary, unlike every other arm's real-BO call), and both
    # otherwise reload their full checkpoint from scratch on every task --
    # the same class of redundant-reload the mi_orpt/warm_*_pool.py modules
    # exist to eliminate elsewhere, just without needing a whole
    # ProcessPoolExecutor here since it's already one process. Memoize the
    # load per (spec, task_set, target) call instead: only pays the load
    # once we know at least one task actually needs it (each function's own
    # dest_csv.exists() skip runs first), and never pays it at all on a
    # fully-resumed sweep.
    optformer_model_provider = None
    llambo_model_provider = None
    if spec.arm == "OptFormer":
        from peptide_experiment import optformer_optimization as _optformer_mod

        # No /epoch_<N> subdirectory for OptFormer checkpoints (arm_specs/main.yaml's
        # own comment); spec.checkpoint_dir is already the right flat path.
        optformer_checkpoint_path = spec.checkpoint_dir
        _optformer_cache: dict = {}

        def optformer_model_provider(_path=optformer_checkpoint_path, _cache=_optformer_cache):
            if "m" not in _cache:
                device = _optformer_mod.resolve_device("auto")
                dtype = _optformer_mod.resolve_dtype("auto", device)
                _cache["m"] = _optformer_mod._load_model_and_tokenizer(_path, device, dtype)
            return _cache["m"]
    elif spec.arm == "LLAMBO":
        from peptide_experiment import llambo_optimization as _llambo_mod

        llambo_checkpoint_path = cfg.base_checkpoint_dir
        _llambo_cache: dict = {}

        def llambo_model_provider(_path=llambo_checkpoint_path, _cache=_llambo_cache):
            if "m" not in _cache:
                device = _llambo_mod.resolve_device("auto")
                dtype = _llambo_mod.resolve_dtype("auto", device)
                _cache["m"] = _llambo_mod._load_model_and_tokenizer(_path, device, dtype)
            return _cache["m"]

    for task_idx in task_ids:
        built = None
        if spec.arm == "STBO":
            # No LLM, no rejection sampling: mutations of the reference
            # peptide, same as MTBO/POGPE/SGPE/OptFormer -- seeded from the
            # SAME shared pool (SHARED_INIT_ARMS) rather than stbo_optimization.py
            # drawing its own fresh, unseeded mutations, so STBO's search is
            # compared from the identical starting point. use_pretrained_vae
            # matches MTBO/POGPE/SGPE below, so every arm's eval-time real-BO
            # starts from the same latent space.
            from peptide_experiment.steps import build_mutation_init

            run_cfg = dataclasses.replace(cfg, use_pretrained_vae=True, init_size=target)
            _seed_shared_mutation_init(run_cfg, spec, task_set, task_idx, target, work_dir)
            init_path, scores_path = build_mutation_init(run_cfg, task_idx, work_dir)
            pool_size, draws_used = target, target
        elif spec.arm == "MTBO":
            # Seeds with exactly `target` guaranteed-feasible random mutations
            # and runs real BO with the pretrained shared surrogate -- a
            # genuinely matched init size and oracle budget vs. every other
            # arm, not a rejection-driven comparison. The checkpoint resolves
            # from spec.run_dir (not cfg's own run_dir), since one arm-spec
            # file can mix arms from different run_dirs. Init pool itself is
            # the SHARED one (SHARED_INIT_ARMS) -- see _seed_shared_mutation_init.
            from peptide_experiment.steps import build_mutation_init

            run_cfg = dataclasses.replace(cfg, use_pretrained_vae=True, init_size=target)
            _seed_shared_mutation_init(run_cfg, spec, task_set, task_idx, target, work_dir)
            init_path, scores_path = build_mutation_init(run_cfg, task_idx, work_dir)
            pool_size, draws_used = target, target
        elif spec.arm == "OptFormer":
            # Its own history-conditioned propose/score loop self-seeds via
            # build_mutation_init (inside run_optformer_bo) -- pre-seeded
            # here with the SAME shared pool MTBO/POGPE/SGPE use, so
            # run_optformer_bo's own build_mutation_init call just finds the
            # files already there and skips regenerating them.
            run_cfg = dataclasses.replace(cfg, init_size=target)
            _seed_shared_mutation_init(run_cfg, spec, task_set, task_idx, target, work_dir)
            pool_size, draws_used = target, target
        elif spec.arm in ("POGPE", "SGPE"):
            # Same seeding as MTBO -- and the same SHARED pool. spec.milestone
            # is reused to mean expert count N here -- a deliberate overload,
            # since N maps directly onto our milestone axis per the paper's
            # framing ("the first 5/10/20 trajectories were used to train the
            # POGPE/SGPE expert models"), rather than being crossed against
            # cfg.milestones.
            from peptide_experiment.steps import build_mutation_init

            run_cfg = dataclasses.replace(cfg, use_pretrained_vae=True, init_size=target)
            _seed_shared_mutation_init(run_cfg, spec, task_set, task_idx, target, work_dir)
            init_path, scores_path = build_mutation_init(run_cfg, task_idx, work_dir)
            pool_size, draws_used = target, target
        elif spec.arm == "LLAMBO":
            # Never fine-tuned -- always the same base checkpoint, so
            # spec.milestone is arbitrary for this arm (see arm_specs/main.yaml).
            # Self-seeds via build_mutation_init inside run_llambo_bo.
            run_cfg = dataclasses.replace(cfg, init_size=target)
            pool_size, draws_used = target, target
        else:
            built = build_bo_pool(dom, cfg, task_idx, raw_dir, work_dir, target=target)
            if built is None:
                continue
            pool_size, draws_used = built.pool_size, built.draws_used
            # dataclasses.replace (not an in-place cfg.init_size= mutation,
            # unlike this branch's prior version) -- cfg is shared across
            # every arm this function gets called for, and every other
            # branch already makes its own run_cfg the same way; mutating
            # the shared cfg in place would leak use_pretrained_vae=True
            # into whichever arm's dataclasses.replace(cfg, ...) call
            # happens to run next. use_pretrained_vae=True matches
            # MTBO/POGPE/SGPE/STBO, so BOLT/ORPT-MI's eval-time real-BO
            # starts from the same latent space as every other arm.
            run_cfg = dataclasses.replace(cfg, use_pretrained_vae=True, init_size=pool_size)

        rejection_rate = 1 - pool_size / draws_used if draws_used else None
        if rejection_rate is not None:
            rejection_rates.append(rejection_rate)

        try:
            if spec.arm == "STBO":
                csv_path = dom.run_bo(
                    run_cfg, task_idx, work_dir, run_id=run_id, stbo=True,
                    init_path=init_path, scores_path=scores_path,
                )
            elif spec.arm == "MTBO":
                surrogate_path = spec.run_dir / "checkpoints" / f"MTBO-{spec.milestone}" / "surrogate_state_dict.pt"
                csv_path = dom.run_bo(
                    run_cfg, task_idx, work_dir, run_id=run_id, init_path=init_path,
                    scores_path=scores_path, pretrained_surrogate_path=surrogate_path,
                )
            elif spec.arm == "OptFormer":
                from peptide_experiment.optformer_optimization import run_optformer_bo

                csv_path = run_optformer_bo(
                    run_cfg, task_idx, work_dir, run_id=run_id, milestone=spec.milestone,
                    checkpoint_path=optformer_checkpoint_path, model_provider=optformer_model_provider,
                )
            elif spec.arm in ("POGPE", "SGPE"):
                from peptide_experiment.gp_expert_transfer import build_sgpe_manifest, fit_sgpe_target_expert

                n_experts = spec.milestone
                base_manifest_path = spec.run_dir / "checkpoints" / f"GPExperts-{n_experts}" / "poe_manifest.json"
                if spec.arm == "SGPE":
                    target_expert_dir = work_dir / f"sgpe_target_expert_task{task_idx:04d}"
                    target_expert_ckpt = fit_sgpe_target_expert(
                        run_cfg, task_idx, target_expert_dir, init_path, scores_path,
                    )
                    manifest_path = work_dir / f"sgpe_manifest_task{task_idx:04d}.json"
                    build_sgpe_manifest(base_manifest_path, target_expert_ckpt, manifest_path)
                else:
                    manifest_path = base_manifest_path
                csv_path = dom.run_bo(
                    run_cfg, task_idx, work_dir, run_id=run_id, init_path=init_path, scores_path=scores_path,
                    surrogate_type="gp_poe", poe_manifest_path=manifest_path, update_e2e=False,
                )
            elif spec.arm == "LLAMBO":
                from peptide_experiment.llambo_optimization import run_llambo_bo

                csv_path = run_llambo_bo(
                    run_cfg, task_idx, work_dir, run_id=run_id, checkpoint_path=None,
                    model_provider=llambo_model_provider,
                )
            else:
                csv_path = dom.run_bo(run_cfg, task_idx, work_dir, run_id=run_id, **built.run_bo_kwargs)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[{run_id} task {task_idx}] FAILED, continuing with rest of sweep: {e}")
            continue
        n_ran_bo += 1
        best_feasible_incumbent = dom.read_best_feasible_incumbent(work_dir, task_idx)
        if best_feasible_incumbent is not None:
            feasible_incumbents.append(best_feasible_incumbent)
        llambo_meta = None
        if spec.arm == "LLAMBO":
            llambo_meta_path = work_dir / f"task_{task_idx:04d}_llambo_meta.json"
            if llambo_meta_path.exists():
                llambo_meta = json.loads(llambo_meta_path.read_text())
        for bo_calls in bo_k_checkpoints(cfg):
            row = {
                "arm": spec.arm, "milestone": spec.milestone, "task_set": task_set, "task_idx": task_idx,
                "target_pool_size": target, "draws_used": draws_used, "rejection_rate": rejection_rate,
                "best_feasible_incumbent": best_feasible_incumbent,
                "bo_calls": bo_calls, "best_mic": dom.best_objective_at_k(cfg, task_idx, csv_path, pool_size, bo_calls),
            }
            if llambo_meta is not None:
                row["llambo_terminated_early"] = llambo_meta["terminated_reason"] != "oracle_budget_reached"
                row["llambo_input_tokens_used"] = llambo_meta["total_input_tokens"]
            bo_rows.append(row)

    coverage_row = {
        "arm": spec.arm, "milestone": spec.milestone, "task_set": task_set, "target_pool_size": target,
        "n_tasks": len(task_ids), "n_ran_bo": n_ran_bo,
        "coverage_rate": n_ran_bo / len(task_ids) if task_ids else None,
        "mean_best_feasible_incumbent": (
            sum(feasible_incumbents) / len(feasible_incumbents) if feasible_incumbents else None
        ),
        "mean_rejection_rate": (
            sum(rejection_rates) / len(rejection_rates) if rejection_rates else None
        ),
    }
    return coverage_row, bo_rows


def compute_rows(
    dom, cfg, specs: list[ArmSpec], task_sets: list[str],
    target_pool_sizes: list[int], limit_tasks: int | None,
) -> tuple[list[dict], list[dict]]:
    coverage_rows, bo_rows = [], []
    for spec in specs:
        for task_set in task_sets:
            # Self-seeding baselines never go through generate_raw (no LLM
            # raw-generation step, no rejection sampling), so this gate
            # doesn't apply to them.
            if spec.arm not in SELF_SEEDING_ARMS and not raw_dir_for(spec, task_set).exists():
                print(f"[eval2.fixed_target_bo] {spec.arm}-{spec.milestone}/{task_set}: "
                      f"no raw generations, run generate_raw first, skipping")
                continue
            task_ids = dom.task_indices(cfg, task_set)
            if limit_tasks is not None:
                task_ids = task_ids[:limit_tasks]
            for target in target_pool_sizes:
                coverage_row, task_bo_rows = run_for_spec_task_set_target(dom, cfg, spec, task_set, target, task_ids)
                coverage_rows.append(coverage_row)
                bo_rows.extend(task_bo_rows)
    return coverage_rows, bo_rows


def summarize_bo_rows(bo_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in bo_rows:
        if row["best_mic"] is None:
            continue
        key = (row["arm"], row["milestone"], row["task_set"], row["target_pool_size"], row["bo_calls"])
        groups.setdefault(key, []).append(row["best_mic"])
    summary = []
    for (arm, milestone, task_set, target, bo_calls), values in sorted(groups.items()):
        summary.append({
            "arm": arm, "milestone": milestone, "task_set": task_set, "target_pool_size": target,
            "bo_calls": bo_calls, "n_tasks_ran_bo": len(values), "mean_best_mic": sum(values) / len(values),
        })
    return summary


def run(
    dom, cfg, specs: list[ArmSpec], task_sets: list[str], out_dir: Path,
    target_pool_sizes: list[int] | None = None, limit_tasks: int | None = None,
) -> None:
    targets = sorted(target_pool_sizes or DEFAULT_TARGET_POOL_SIZES)
    coverage_rows, bo_rows = compute_rows(dom, cfg, specs, task_sets, targets, limit_tasks)
    write_csv(coverage_rows, out_dir / "fixed_target_bo_coverage.csv", fieldnames=COVERAGE_FIELDS)
    write_csv(bo_rows, out_dir / "per_task_fixed_target_bo.csv", fieldnames=PER_TASK_FIELDS)
    write_csv(summarize_bo_rows(bo_rows), out_dir / "summary_fixed_target_bo.csv", fieldnames=SUMMARY_FIELDS)

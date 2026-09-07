"""(2.2) Build a *fixed-size* pool via real rejection sampling (no synthetic
top-up), for each of several target pool sizes, then run real BO on it --
comparing models on equal footing at a chosen initialization-pool size. This
is the compute engine behind fig:main-bo (paper/experiments.tex
sec:main-results, "Full-budget optimization") and, at a fixed bo_calls/
target_pool_size slice, fig:scaling.

If a task's already-generated raw proposals don't contain `target` feasible
unique candidates, that task is skipped for that target (no on-demand extra
sampling) and counted in a coverage_rate metric -- see
common.build_bo_pool(target=...).

Reports coverage separately from BO performance: an arm/milestone that only
clears a given target for a few tasks is a real finding (a naive-DPO arm
paying for its own constraint violations in reduced coverage), not something
to silently average away.

STBO/MTBO/OptFormer/POGPE/SGPE/LLAMBO are peptide-only baselines (no
query_plan analog exists yet, per query_plan_experiment/README.md's own
"## Scope" note) -- self-seeding arms that build their own init pool
directly here rather than going through generate_raw_proposals.py/
domain.build_bo_pool.

Reads raw generations via common.raw_dir_for(spec, task_set) -- run
generate_raw_proposals.py first for any (arm, milestone, task_set) not yet
covered. Each covered (task, target) launches one real BO run -- start
with --limit-tasks and one target size before a full sweep.

Usage (run from the BOLT repo root):
    python experiments/eval/fixed_target_rejection_bo.py \\
        --domain peptide \\
        --config peptide_experiment/configs/peptide_poc20_bolt.yaml \\
        --manifest experiments/eval/manifests/main_bolt_vs_orpt_mi.yaml \\
        --target-pool-sizes 10 --limit-tasks 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

from common import ModelSpec, bo_k_checkpoints, build_bo_pool, load_manifest, raw_dir_for, read_best_feasible_incumbent, write_csv
from domains import DOMAINS, Domain

PEPTIDE_ONLY_BASELINE_ARMS = ("STBO", "MTBO", "OptFormer", "POGPE", "SGPE", "LLAMBO")

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


def run_for_spec_task_set_target(
    domain: Domain, cfg, spec: ModelSpec, task_set: str, target: int, task_ids: list,
) -> tuple[dict, list[dict]]:
    if spec.arm in PEPTIDE_ONLY_BASELINE_ARMS:
        assert domain.name == "peptide", f"{spec.arm} baseline not implemented for domain={domain.name!r} yet"

    raw_dir = raw_dir_for(spec, task_set)
    work_dir = spec.run_dir / "eval_fixed_target_bo" / task_set / f"{spec.arm}-{spec.milestone}__target{target}"
    run_id = f"fixedtarget-{task_set}-{spec.arm}-{spec.milestone}-t{target}"

    n_ran_bo = 0
    rejection_rates: list[float] = []
    feasible_incumbents: list[float] = []
    bo_rows: list[dict] = []
    for task_idx in task_ids:
        if spec.arm == "STBO":
            # No LLM, no rejection sampling: stbo_optimization.py builds its
            # own guaranteed-size init pool internally (mutations of the
            # reference peptide), same primitive
            # peptide_experiment/steps.py::build_mutation_init also uses --
            # just needs cfg.init_size pinned to target, no init_path/
            # scores_path from us at all (see steps.py::run_bo's stbo=True
            # branch, and heldout_eval.py's identical call shape).
            run_cfg = dataclasses.replace(cfg, init_size=target)
            pool_size, draws_used = target, target
        elif spec.arm == "MTBO":
            # No LLM, so no rejection sampling: seeds with exactly `target`
            # guaranteed-feasible random mutations (same primitive STBO
            # uses) and runs real BO with the pretrained shared surrogate --
            # a genuinely matched init size + oracle budget vs. every other
            # arm, not a rejection-driven comparison. Checkpoint resolved
            # from spec.run_dir (not cfg's own run_dir), since one manifest
            # can mix arms from different run_dirs.
            from peptide_experiment.steps import build_mutation_init

            run_cfg = dataclasses.replace(cfg, use_pretrained_vae=True, init_size=target)
            init_path, scores_path = build_mutation_init(run_cfg, task_idx, work_dir)
            pool_size, draws_used = target, target
        elif spec.arm == "OptFormer":
            # Same idea: its own history-conditioned propose/score loop
            # self-seeds via build_mutation_init (inside run_optformer_bo),
            # genuinely re-run per target -- no reuse across target_pool_size
            # panels.
            run_cfg = dataclasses.replace(cfg, init_size=target)
            pool_size, draws_used = target, target
        elif spec.arm in ("POGPE", "SGPE"):
            # No LLM, no rejection sampling (same reasoning as MTBO): seeds
            # with exactly `target` guaranteed-feasible random mutations.
            # spec.milestone is reused to mean expert count N here (a
            # deliberate, minor overload of the shared ModelSpec.milestone
            # field -- N maps directly onto our milestone axis per the
            # paper's own framing, "the first 5/10/20 trajectories were used
            # to train the POGPE/SGPE expert models", not crossed against
            # cfg.milestones).
            from peptide_experiment.steps import build_mutation_init

            run_cfg = dataclasses.replace(cfg, use_pretrained_vae=True, init_size=target)
            init_path, scores_path = build_mutation_init(run_cfg, task_idx, work_dir)
            pool_size, draws_used = target, target
        elif spec.arm == "LLAMBO":
            # No LLM checkpoint per-milestone (LLAMBO is never fine-tuned --
            # always the same base, unmodified checkpoint -- spec.milestone
            # is unused/arbitrary for this arm; see the manifest's own
            # comment). Self-seeds via build_mutation_init inside
            # run_llambo_bo, same as OptFormer.
            run_cfg = dataclasses.replace(cfg, init_size=target)
            pool_size, draws_used = target, target
        else:
            built = build_bo_pool(domain, cfg, task_idx, raw_dir, work_dir, target=target)
            if built is None:
                continue
            pool_size, draws_used = built.pool_size, built.draws_used
            cfg.init_size = pool_size  # run_bo reads cfg.init_size; must match target exactly

        rejection_rate = 1 - pool_size / draws_used if draws_used else None
        if rejection_rate is not None:
            rejection_rates.append(rejection_rate)

        try:
            if spec.arm == "STBO":
                csv_path = domain.run_bo(run_cfg, task_idx, work_dir, run_id=run_id, stbo=True)
            elif spec.arm == "MTBO":
                surrogate_path = spec.run_dir / "checkpoints" / f"MTBO-{spec.milestone}" / "surrogate_state_dict.pt"
                csv_path = domain.run_bo(
                    run_cfg, task_idx, work_dir, run_id=run_id, init_path=init_path,
                    scores_path=scores_path, pretrained_surrogate_path=surrogate_path,
                )
            elif spec.arm == "OptFormer":
                from peptide_experiment.optformer_optimization import run_optformer_bo

                checkpoint_path = (
                    spec.run_dir / "checkpoints" / f"OptFormer-{spec.milestone}" / f"epoch_{cfg.optformer_epochs - 1}"
                )
                csv_path = run_optformer_bo(
                    run_cfg, task_idx, work_dir, run_id=run_id, milestone=spec.milestone,
                    checkpoint_path=checkpoint_path,
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
                csv_path = domain.run_bo(
                    run_cfg, task_idx, work_dir, run_id=run_id, init_path=init_path, scores_path=scores_path,
                    surrogate_type="gp_poe", poe_manifest_path=manifest_path, update_e2e=False,
                )
            elif spec.arm == "LLAMBO":
                from peptide_experiment.llambo_optimization import run_llambo_bo

                csv_path = run_llambo_bo(run_cfg, task_idx, work_dir, run_id=run_id, checkpoint_path=None)
            else:
                csv_path = domain.run_bo(cfg, task_idx, work_dir, run_id=run_id, **built.run_bo_kwargs)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[{run_id} task {task_idx}] FAILED, continuing with rest of sweep: {e}")
            continue
        n_ran_bo += 1
        best_feasible_incumbent = read_best_feasible_incumbent(work_dir, task_idx) if domain.name == "peptide" else None
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
                "bo_calls": bo_calls, "best_mic": domain.best_objective_at_k(cfg, task_idx, csv_path, pool_size, bo_calls),
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
    domain: Domain, cfg, specs: list[ModelSpec], task_set_names: list[str],
    target_pool_sizes: list[int], limit_tasks: int | None,
) -> tuple[list[dict], list[dict]]:
    coverage_rows, bo_rows = [], []
    for spec in specs:
        for task_set in task_set_names:
            # Self-seeding baselines never go through generate_raw_proposals.py (no
            # LLM raw-generation step, no rejection sampling) -- they build a target-sized
            # init pool directly inside run_for_spec_task_set_target(), so this
            # raw-generation gate doesn't apply.
            if spec.arm not in PEPTIDE_ONLY_BASELINE_ARMS and not raw_dir_for(spec, task_set).exists():
                print(f"[fixed_target_rejection_bo] {spec.arm}-{spec.milestone}/{task_set}: "
                      f"no raw generations, run generate_raw_proposals.py first, skipping")
                continue
            task_ids = domain.task_indices(cfg, task_set)
            if limit_tasks is not None:
                task_ids = task_ids[:limit_tasks]
            for target in target_pool_sizes:
                coverage_row, task_bo_rows = run_for_spec_task_set_target(domain, cfg, spec, task_set, target, task_ids)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="peptide", choices=sorted(DOMAINS))
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-sets", default="heldout")
    parser.add_argument(
        "--target-pool-sizes",
        default=",".join(str(t) for t in DEFAULT_TARGET_POOL_SIZES),
        help="Comma-separated fixed init-pool sizes to compare models at",
    )
    parser.add_argument("--limit-tasks", type=int, default=None, help="Smoke-test: first N tasks per task_set only")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/<manifest stem>")
    args = parser.parse_args()

    domain = DOMAINS[args.domain]
    cfg = domain.load_config(args.config)
    specs = load_manifest(args.manifest)
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]
    target_pool_sizes = sorted(int(t.strip()) for t in args.target_pool_sizes.split(",") if t.strip())

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(__file__).resolve().parent / "results" / Path(args.manifest).stem
    )

    coverage_rows, bo_rows = compute_rows(domain, cfg, specs, task_set_names, target_pool_sizes, args.limit_tasks)
    write_csv(coverage_rows, out_dir / "fixed_target_bo_coverage.csv", fieldnames=COVERAGE_FIELDS)
    write_csv(bo_rows, out_dir / "per_task_fixed_target_bo.csv", fieldnames=PER_TASK_FIELDS)
    write_csv(summarize_bo_rows(bo_rows), out_dir / "summary_fixed_target_bo.csv", fieldnames=SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

"""Run an actual BO trial per task, seeded by *exactly* the candidates that
survived real rejection sampling -- discarding infeasible raw generations
outright, never topping the pool back up with synthetic reference-mutations
the way steps.py::_ensure_constraint_feasible does for the paper-fidelity
pipeline, and never truncating/padding to a fixed target size either. The
feasible-pool size that results (anywhere from a handful to ~1400 in this
experiment's data) is itself part of the finding -- forcing every task to
the same init size would hide the real effect being measured: an arm/
milestone whose proposals violate the constraint more ends up with fewer,
not just worse, initialization points, and that legitimately feeds into
worse BO outcomes downstream. Only a small absolute floor is enforced
(--min-feasible, default 5), to dodge the documented LOLBO trust-region
hang when the init pool is (near-)empty -- see
steps.py::_ensure_constraint_feasible's docstring.

This closes a gap in this directory: measure_violation_rate.py quantifies
how often BOLT-<m>/ORPT-<m> proposals violate the similarity constraint, but
nothing here (or anywhere else in peptide_experiment/) reports what actually
happens to BO's optimization outcome once you act on that violation rate by
rejecting infeasible proposals. run_heldout_eval() does call run_bo(), but
its init pool goes through the patch-based _ensure_constraint_feasible, so
it under-reports what a real reject-and-discard policy would face.

Reuses already-generated raw generations from run_trainset_eval.py's
trainset_eval/ output and heldout_eval.run_init_only_eval()'s
heldout20/init_only/ output (via measure_violation_rate.task_dir_for) --
this script does NOT invoke sampling_transformers.py itself, so run those
first for whichever (arm, milestone, task_set) combinations you want covered
here.

Each covered task launches one real LOLBO run (up to cfg.oracle_budget
oracle calls) -- expensive. Start with a small --arms/--task-sets/milestone
subset before sweeping everything.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/run_rejection_sampled_bo.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))

from peptide_experiment.config import ExperimentConfig, load_config  # noqa: E402
from peptide_experiment.steps import run_bo  # noqa: E402
from apex_oracle import apex_wrapper  # noqa: E402
from apex_oracle.refseqs import REFERENCE_SEQUENCE  # noqa: E402

from run_trainset_eval import ARM_CHECKPOINT_FN  # noqa: E402
from measure_violation_rate import load_raw_generations, similarity, task_dir_for, task_sets  # noqa: E402

DEFAULT_MIN_FEASIBLE = 5


def real_rejection_sample(cfg: ExperimentConfig, task_idx: int, raw_task_dir: Path) -> list[str]:
    """Unique, order-preserved, similarity-feasible raw generations only --
    no synthetic top-up, no truncation. Size varies task to task."""
    reference = REFERENCE_SEQUENCE[task_idx]
    sequences = load_raw_generations(raw_task_dir, task_idx)
    feasible, seen = [], set()
    for seq in sequences:
        if seq in seen:
            continue
        seen.add(seq)
        if similarity(seq, reference) >= cfg.similarity_threshold:
            feasible.append(seq)
    return feasible


def build_rejection_sampled_init(
    cfg: ExperimentConfig, task_idx: int, raw_task_dir: Path, work_dir: Path, min_feasible: int
) -> tuple[Path, Path, int] | None:
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    if init_path.exists() and scores_path.exists():
        pool_size = sum(1 for line in init_path.read_text().splitlines() if line.strip())
        return init_path, scores_path, pool_size

    feasible = real_rejection_sample(cfg, task_idx, raw_task_dir)
    if len(feasible) < min_feasible:
        print(
            f"[task {task_idx}] only {len(feasible)} feasible unique candidates survived real "
            f"rejection sampling in {raw_task_dir} (< --min-feasible={min_feasible}); skipping BO "
            "for this task (no synthetic top-up, no padding)"
        )
        return None

    scores = list(-apex_wrapper(feasible)[:, 0])

    work_dir.mkdir(parents=True, exist_ok=True)
    init_path.write_text("\n".join(feasible) + "\n")
    scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")
    return init_path, scores_path, len(feasible)


def run_rejection_sampled_bo(
    cfg: ExperimentConfig,
    arm_prefixes: list[str],
    task_set_names: list[str],
    min_feasible: int,
    milestones: list[int] | None = None,
) -> None:
    sets = task_sets(cfg)
    for milestone in milestones if milestones is not None else cfg.milestones:
        for arm in arm_prefixes:
            checkpoint_dir = ARM_CHECKPOINT_FN[arm](cfg, milestone)
            if not checkpoint_dir.exists():
                print(f"[rejection_sampled_bo] {arm}-{milestone}: checkpoint not found, skipping")
                continue
            for task_set in task_set_names:
                raw_task_dir = task_dir_for(cfg, task_set, arm, milestone)
                if not raw_task_dir.exists():
                    print(
                        f"[rejection_sampled_bo] {arm}-{milestone}/{task_set}: no raw generations at "
                        f"{raw_task_dir} -- run run_trainset_eval.py / heldout_eval init_only_eval first, skipping"
                    )
                    continue

                work_dir = cfg.run_dir / "constraint_violation_bo" / task_set / f"{arm}-{milestone}"
                run_id = f"rejsample-{task_set}-{arm}-{milestone}"
                n_ok, n_skipped_infeasible, n_failed = 0, 0, 0
                for task_idx in sets[task_set]:
                    try:
                        built = build_rejection_sampled_init(cfg, task_idx, raw_task_dir, work_dir, min_feasible)
                        if built is None:
                            n_skipped_infeasible += 1
                            continue
                        init_path, scores_path, pool_size = built
                        # run_bo reads cfg.init_size to pass --num_initialization_points;
                        # it must match this task's actual (variable) pool size exactly.
                        cfg.init_size = pool_size
                        run_bo(
                            cfg,
                            task_idx,
                            work_dir,
                            run_id=run_id,
                            init_path=init_path,
                            scores_path=scores_path,
                        )
                        n_ok += 1
                    except (subprocess.CalledProcessError, RuntimeError) as e:
                        print(f"[{run_id} task {task_idx}] FAILED, continuing with rest of sweep: {e}")
                        n_failed += 1
                print(
                    f"[{run_id}] done: {n_ok} ran BO, {n_skipped_infeasible} skipped "
                    f"(fewer than {min_feasible} feasible candidates), {n_failed} failed, "
                    f"out of {len(sets[task_set])} tasks"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arms", default="BOLT,ORPT")
    parser.add_argument("--task-sets", default="trainset,heldout20")
    parser.add_argument(
        "--milestones",
        default=None,
        help="Comma-separated subset of cfg.milestones to run (default: all of them). "
        "Use this to sanity-check on one cheap milestone before committing to a full sweep.",
    )
    parser.add_argument(
        "--min-feasible",
        type=int,
        default=DEFAULT_MIN_FEASIBLE,
        help="Skip a task if real rejection sampling finds fewer than this many feasible "
        "unique candidates (avoids the documented LOLBO trust-region hang on a near-empty "
        f"init pool). Default {DEFAULT_MIN_FEASIBLE}.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm_prefixes = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_prefixes:
        assert a in ARM_CHECKPOINT_FN, f"unknown arm prefix {a!r}, expected one of {list(ARM_CHECKPOINT_FN)}"
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]
    for t in task_set_names:
        assert t in ("trainset", "heldout20"), f"unknown task set {t!r}"
    milestones = (
        [int(m.strip()) for m in args.milestones.split(",") if m.strip()]
        if args.milestones is not None
        else None
    )

    cfg.ensure_dirs()
    run_rejection_sampled_bo(cfg, arm_prefixes, task_set_names, args.min_feasible, milestones)


if __name__ == "__main__":
    main()

"""Clean split of Fixed-Budget's rejection rate into its component signals,
at a literal fixed sampling budget N (same N for every arm/milestone/task,
unlike fixed_budget_rejection_bo.py's adaptive draws_used which stops early
once enough unique candidates are found -- see that script's docstring):

- violation_rate: of the first N raw draws (duplicates included, in
  generation order), the fraction that fail the similarity constraint
  (similarity(seq, reference) < cfg.similarity_threshold). This is the
  "how often does the LLM itself propose something that breaks the rule"
  number, uncontaminated by how repetitive it is.
- duplication_rate: of the same N draws, 1 - (#unique / N). Diversity/mode
  collapse, reported separately rather than folded into rejection rate.
- pool_rejection_rate: of the same N draws, 1 - (#accepted / N), where
  "accepted" replicates feasible_pool_with_draw_counts()'s actual pool-build
  rule (accept iff this is the sequence's first occurrence AND it's
  feasible) -- i.e. this is what real pool construction (Experiments 2-3)
  throws away, at the same fixed, unbiased N. Not simply violation_rate +
  duplication_rate (a duplicate of an already-violating sequence isn't
  double-counted), so it's tracked as its own pass rather than derived.

feasible_pool_with_draw_counts() (used by build_bo_pool() elsewhere in this
package) skips a duplicate seq without ever constraint-checking it, so its
"rejected" count silently mixes duplication and violation together --
violation_rate/duplication_rate split that apart; pool_rejection_rate is the
reunified number, just measured at a fixed N instead of an adaptive one.

No new sampling: reads the same raw jsonls generate_raw_proposals.py already
wrote, just truncated to the first N draws per task.

Usage (run from the BOLT repo root):
    python experiments/eval/violation_duplication_rate.py \\
        --config peptide_experiment/configs/peptide_poc20_bolt.yaml \\
        --manifest experiments/eval/manifests/poc20_four_arm.yaml \\
        --n 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from common import ModelSpec, load_manifest, load_raw_generations, raw_dir_for, similarity, task_indices, write_csv
from plot_common import MUTED_TEXT, arm_colors, sorted_arms, style_axis

from apex_oracle.refseqs import REFERENCE_SEQUENCE
from peptide_experiment.config import ExperimentConfig, load_config

PER_TASK_FIELDS = [
    "arm", "milestone", "task_set", "task_idx", "n", "violation_rate", "duplication_rate", "pool_rejection_rate",
]
SUMMARY_FIELDS = [
    "arm", "milestone", "task_set", "n", "n_tasks",
    "mean_violation_rate", "mean_duplication_rate", "mean_pool_rejection_rate",
]
TASK_SET_LABEL = {"heldout": "held-out peptides", "trainset": "trained peptides"}


def rates_for_task(cfg: ExperimentConfig, task_idx: int, raw_dir: Path, n: int) -> tuple[float, float, float] | None:
    sequences = load_raw_generations(raw_dir, task_idx)
    if len(sequences) < n:
        return None
    draws = sequences[:n]
    reference = REFERENCE_SEQUENCE[task_idx]

    n_violations = 0
    n_accepted = 0
    seen: set[str] = set()
    for seq in draws:
        feasible = similarity(seq, reference) >= cfg.similarity_threshold
        if not feasible:
            n_violations += 1
        if seq not in seen and feasible:
            n_accepted += 1
        seen.add(seq)
    n_unique = len(seen)

    return n_violations / n, 1 - n_unique / n, 1 - n_accepted / n


def compute_rows(
    cfg: ExperimentConfig, specs: list[ModelSpec], task_set_names: list[str], n: int, limit_tasks: int | None,
) -> list[dict]:
    rows = []
    for spec in specs:
        for task_set in task_set_names:
            raw_dir = raw_dir_for(spec, task_set)
            if not raw_dir.exists():
                print(f"[violation_duplication_rate] {spec.arm}-{spec.milestone}/{task_set}: "
                      f"no raw generations, run generate_raw_proposals.py first, skipping")
                continue
            task_ids = task_indices(cfg, task_set)
            if limit_tasks is not None:
                task_ids = task_ids[:limit_tasks]
            for task_idx in task_ids:
                result = rates_for_task(cfg, task_idx, raw_dir, n)
                if result is None:
                    print(f"[violation_duplication_rate] {spec.arm}-{spec.milestone}/{task_set}/task{task_idx}: "
                          f"fewer than n={n} raw draws available, skipping")
                    continue
                violation_rate, duplication_rate, pool_rejection_rate = result
                rows.append({
                    "arm": spec.arm, "milestone": spec.milestone, "task_set": task_set, "task_idx": task_idx,
                    "n": n, "violation_rate": violation_rate, "duplication_rate": duplication_rate,
                    "pool_rejection_rate": pool_rejection_rate,
                })
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["arm"], row["milestone"], row["task_set"], row["n"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (arm, milestone, task_set, n), group in sorted(groups.items()):
        summary.append({
            "arm": arm, "milestone": milestone, "task_set": task_set, "n": n, "n_tasks": len(group),
            "mean_violation_rate": sum(r["violation_rate"] for r in group) / len(group),
            "mean_duplication_rate": sum(r["duplication_rate"] for r in group) / len(group),
            "mean_pool_rejection_rate": sum(r["pool_rejection_rate"] for r in group) / len(group),
        })
    return summary


def plot_metric(rows: list[dict], task_set: str, metric: str, label: str, out_path: Path, n: int) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    sub = df[df["task_set"] == task_set]
    if sub.empty:
        print(f"[violation_duplication_rate] no data for task_set={task_set}, skipping {out_path}")
        return
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm]
        stats = arm_sub.groupby("milestone")[metric].agg(["mean", "std"]).reindex(milestones)
        ax.plot(milestones, stats["mean"], color=colors[arm], linewidth=2, marker="o", markersize=8, label=arm)
        ax.fill_between(
            milestones,
            stats["mean"] - stats["std"].fillna(0),
            stats["mean"] + stats["std"].fillna(0),
            color=colors[arm],
            alpha=0.15,
            linewidth=0,
        )
    ax.set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)
    ax.set_ylabel(label, color=MUTED_TEXT)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(f"{label} vs. #tasks trained (N={n} raw proposals, {TASK_SET_LABEL[task_set]})", color="#0b0b0b")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-sets", default="trainset,heldout")
    parser.add_argument("--n", type=int, default=500, help="Fixed sampling budget (raw draws per task)")
    parser.add_argument("--limit-tasks", type=int, default=None, help="Smoke-test: first N tasks per task_set only")
    parser.add_argument("--arms", default=None, help="Comma-separated subset of arms to compute/plot (default: all present)")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/<manifest stem>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    specs = load_manifest(args.manifest)
    if args.arms:
        wanted = {a.strip().upper() for a in args.arms.split(",")}
        specs = [s for s in specs if s.arm.upper() in wanted]
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(__file__).resolve().parent / "results" / Path(args.manifest).stem
    )

    rows = compute_rows(cfg, specs, task_set_names, args.n, args.limit_tasks)
    write_csv(rows, out_dir / "per_task_violation_duplication.csv", fieldnames=PER_TASK_FIELDS)
    write_csv(summarize_rows(rows), out_dir / "summary_violation_duplication.csv", fieldnames=SUMMARY_FIELDS)

    plots_dir = out_dir / "plots"
    for task_set in task_set_names:
        plot_metric(
            rows, task_set, "violation_rate", "constraint violation rate",
            plots_dir / f"violation_bymilestone_{task_set}.png", n=args.n,
        )
        plot_metric(
            rows, task_set, "duplication_rate", "duplication rate",
            plots_dir / f"duplication_bymilestone_{task_set}.png", n=args.n,
        )
        plot_metric(
            rows, task_set, "pool_rejection_rate", "pool rejection rate",
            plots_dir / f"pool_rejection_bymilestone_{task_set}.png", n=args.n,
        )


if __name__ == "__main__":
    main()

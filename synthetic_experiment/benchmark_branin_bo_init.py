"""Benchmark Branin GP-BO convergence as the initial design size changes.

This script is intentionally independent of the BOLT/ORPT experiment code.
It implements the task-varying Branin function, uniform initial designs, a
fixed-kernel RBF Gaussian process, and GP-UCB acquisition locally.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np


BOUNDS = np.asarray([[-5.0, 10.0], [0.0, 15.0]], dtype=np.float64)


def branin(points: np.ndarray, task_t: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x1, x2 = points[..., 0], points[..., 1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - task_t) * np.cos(x1) + s


def uniform_points(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n, 2))


def scale_points(points: np.ndarray) -> np.ndarray:
    return (points - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def gp_ucb_candidate(
    x: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
    candidate_pool_size: int,
    lengthscale: float,
    beta: float,
) -> np.ndarray:
    """Minimize a GP lower-confidence bound over a fresh uniform pool."""
    candidates = uniform_points(rng, candidate_pool_size)
    sx, sc = scale_points(x), scale_points(candidates)
    sq_xx = np.sum((sx[:, None, :] - sx[None, :, :]) ** 2, axis=-1)
    sq_xc = np.sum((sx[:, None, :] - sc[None, :, :]) ** 2, axis=-1)
    kxx = np.exp(-0.5 * sq_xx / lengthscale**2)
    kxc = np.exp(-0.5 * sq_xc / lengthscale**2)

    mean_y = float(values.mean())
    scale_y = max(float(values.std()), 1.0)
    covariance = kxx + 1e-6 * np.eye(len(x))
    try:
        chol = np.linalg.cholesky(covariance)
        normalized_y = (values - mean_y) / scale_y
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, normalized_y))
        solved = np.linalg.solve(chol, kxc)
    except np.linalg.LinAlgError:
        return candidates[0]

    posterior_mean = mean_y + scale_y * (kxc.T @ alpha)
    posterior_variance = np.maximum(1.0 - np.sum(solved**2, axis=0), 1e-12)
    lower_confidence_bound = posterior_mean - beta * scale_y * np.sqrt(posterior_variance)
    return candidates[int(np.argmin(lower_confidence_bound))]


def first_hit(best_regret: list[float], threshold: float) -> int | None:
    return next((index for index, regret in enumerate(best_regret) if regret <= threshold), None)


def run_trial(
    init_size: int,
    task_t: float,
    seed: int,
    max_bo_steps: int,
    candidate_pool_size: int,
    lengthscale: float,
    beta: float,
    strictest_threshold: float,
    minimum_steps: int = 0,
) -> tuple[list[float], float]:
    rng = np.random.default_rng(seed)
    x = uniform_points(rng, init_size)
    values = branin(x, task_t)
    optimum = 10.0 * task_t
    best_regret = [max(float(values.min() - optimum), 0.0)]

    for _ in range(max_bo_steps):
        point = gp_ucb_candidate(
            x, values, rng, candidate_pool_size, lengthscale, beta
        )
        x = np.vstack([x, point])
        values = np.append(values, branin(point, task_t))
        best_regret.append(max(float(values.min() - optimum), 0.0))
        if best_regret[-1] <= strictest_threshold and len(best_regret) - 1 >= minimum_steps:
            break
    return best_regret, float(best_regret[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-sizes", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100])
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--max-bo-steps", type=int, default=300)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1e-2, 5e-3, 1e-3])
    parser.add_argument(
        "--report-bo-steps",
        type=int,
        nargs="+",
        default=[30],
        help="Report regret after exactly these many BO calls.",
    )
    parser.add_argument("--lengthscale", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/branin_bo_init_benchmark"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.init_sizes) < 2 or min(args.thresholds) <= 0:
        raise ValueError("init sizes must be >=2 and thresholds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_values = np.linspace(0.0, 1.0, args.num_tasks)
    strictest = min(args.thresholds)
    if min(args.report_bo_steps) < 0 or max(args.report_bo_steps) > args.max_bo_steps:
        raise ValueError("report BO steps must be between 0 and max-bo-steps")
    minimum_steps = max(args.report_bo_steps)
    records: list[dict] = []
    started = time.time()

    for init_size in args.init_sizes:
        for seed_index in range(args.num_seeds):
            for task_index, task_t in enumerate(task_values):
                trial_seed = args.seed + init_size * 100_000 + seed_index * 10_000 + task_index
                regrets, final_regret = run_trial(
                    init_size,
                    float(task_t),
                    trial_seed,
                    args.max_bo_steps,
                    args.candidate_pool_size,
                    args.lengthscale,
                    args.beta,
                    strictest,
                    minimum_steps,
                )
                row = {
                    "init_size": init_size,
                    "task_index": task_index,
                    "task_t": float(task_t),
                    "seed_index": seed_index,
                    "seed": trial_seed,
                    "final_regret": final_regret,
                    "bo_steps_run": len(regrets) - 1,
                }
                for threshold in args.thresholds:
                    hit = first_hit(regrets, threshold)
                    key = f"hit_{threshold:g}"
                    row[key] = hit
                    row[f"total_points_{threshold:g}"] = None if hit is None else init_size + hit
                for step in args.report_bo_steps:
                    row[f"regret_at_bo_{step}"] = regrets[step]
                records.append(row)
        print(f"completed init_size={init_size}; elapsed={time.time() - started:.1f}s", flush=True)

    detail_path = args.output_dir / "trials.csv"
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_rows = []
    for init_size in args.init_sizes:
        subset = [row for row in records if row["init_size"] == init_size]
        for threshold in args.thresholds:
            key = f"total_points_{threshold:g}"
            points = np.asarray([row[key] for row in subset if row[key] is not None], dtype=float)
            summary_rows.append({
                "init_size": init_size,
                "threshold": threshold,
                "num_trials": len(subset),
                "num_converged": len(points),
                "success_rate": len(points) / len(subset),
                "mean_total_points": float(points.mean()) if len(points) else math.nan,
                "median_total_points": float(np.median(points)) if len(points) else math.nan,
                "p75_total_points": float(np.quantile(points, 0.75)) if len(points) else math.nan,
            })

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    checkpoint_rows = []
    for init_size in args.init_sizes:
        subset = [row for row in records if row["init_size"] == init_size]
        for step in args.report_bo_steps:
            regrets = np.asarray([row[f"regret_at_bo_{step}"] for row in subset], dtype=float)
            checkpoint_rows.append({
                "init_size": init_size,
                "bo_steps": step,
                "total_points": init_size + step,
                "num_trials": len(regrets),
                "mean_regret": float(regrets.mean()),
                "std_regret": float(regrets.std(ddof=1)),
                "variance_regret": float(regrets.var(ddof=1)),
                "median_regret": float(np.median(regrets)),
                "p25_regret": float(np.quantile(regrets, 0.25)),
                "p75_regret": float(np.quantile(regrets, 0.75)),
            })

    checkpoint_path = args.output_dir / "regret_by_bo_step.csv"
    with checkpoint_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}\n")
    print(f"Saved: {checkpoint_path}\n")
    for row in summary_rows:
        print(
            f"init={row['init_size']:>3} threshold={row['threshold']:<7g} "
            f"success={row['num_converged']:>2}/{row['num_trials']} "
            f"mean_total={row['mean_total_points']:.1f} "
            f"median_total={row['median_total_points']:.1f} "
            f"p75_total={row['p75_total_points']:.1f}"
        )
    print("\nRegret at fixed BO steps:")
    for row in checkpoint_rows:
        print(
            f"init={row['init_size']:>3} bo={row['bo_steps']:>3} "
            f"total={row['total_points']:>3} mean={row['mean_regret']:.6g} "
            f"std={row['std_regret']:.6g} variance={row['variance_regret']:.6g} "
            f"median={row['median_regret']:.6g}"
        )


if __name__ == "__main__":
    main()

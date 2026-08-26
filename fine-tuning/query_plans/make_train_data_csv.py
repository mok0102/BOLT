"""Turn cumulative per-workload BO trajectory CSVs into SFT training data for
the query-plan domain: one row per selected top-N trajectory candidate, in
the {task, train_x} schema generate_openai_ft_data.py already expects (it
looks up each workload's SQL text itself via get_task_description()/
extract_task_id() -- this script only needs to supply the workload name and
the target completion text).

No feasibility filter is needed here (unlike peptide's own
make_train_data_csv.py, which filters to similarity-constraint-feasible
candidates first) -- the query-plan domain has no such constraint, confirmed
directly from the paper (see
imp_plan/02_query_plan_reimplementation_plan.md). Only censored (timed-out/
failed) rows are excluded: their train_y is a lower-bound proxy score (see
your_tasks/your_objective_functions.py::DatabaseObjective), not a genuine
completed-query runtime, so they'd bias "top scoring" selection toward
candidates that just happened to get a lenient censored-penalty value rather
than actually running fast.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_TOP_N = 50


def collect_top_rows(input_csv: Path, top_n: int) -> tuple[list[tuple[float, str]], int, int]:
    """Returns (top_n (score, train_x_repr) pairs by score descending,
    rows_skipped as unparseable, rows_skipped as censored)."""
    rows_skipped = 0
    rows_censored = 0
    scored_rows: list[tuple[float, str]] = []

    with input_csv.open(newline="") as f_in:
        reader = csv.DictReader(f_in)
        required_columns = {"train_x", "train_y"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Expected input CSV to contain columns {sorted(required_columns)}: {input_csv}")

        for row in reader:
            train_x = (row.get("train_x") or "").strip()
            if not train_x:
                rows_skipped += 1
                continue
            try:
                score = float(row["train_y"])
            except ValueError:
                rows_skipped += 1
                continue

            censored = str(row.get("censoring", "0")).strip() not in ("0", "0.0", "")
            if censored:
                rows_censored += 1
                continue

            scored_rows.append((score, train_x))

    top_rows = sorted(scored_rows, reverse=True)[:top_n]
    return top_rows, rows_skipped, rows_censored


def make_train_data_csv(
    input_csvs: list[Path],
    workload_names: list[str],
    output_csv: Path,
    top_n: int,
) -> None:
    if len(input_csvs) != len(workload_names):
        raise ValueError(
            f"Expected one workload name per input CSV, got {len(input_csvs)} "
            f"input CSVs and {len(workload_names)} workload names."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows_written = 0

    with output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["task", "train_x"])
        writer.writeheader()

        for input_csv, workload_name in zip(input_csvs, workload_names):
            top_rows, rows_skipped, rows_censored = collect_top_rows(input_csv, top_n)
            for _, train_x in top_rows:
                writer.writerow({"task": workload_name, "train_x": train_x})

            total_rows_written += len(top_rows)
            print(f"Input CSV: {input_csv} (workload {workload_name})")
            print(f"  Rows written: {len(top_rows)}")
            print(f"  Rows skipped (unparseable): {rows_skipped}")
            print(f"  Rows skipped (censored): {rows_censored}")
            print(f"  Top score: {top_rows[0][0] if top_rows else 'n/a'}")
            print(f"  Bottom included score: {top_rows[-1][0] if top_rows else 'n/a'}")

    print(f"Wrote CSV: {output_csv}")
    print(f"Total rows written: {total_rows_written}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an SFT training CSV for generate_openai_ft_data.py from query-plan BO trajectory data."
    )
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--workload-name", nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_train_data_csv(
        input_csvs=args.input_csv,
        workload_names=args.workload_name,
        output_csv=args.output_csv,
        top_n=args.top_n,
    )

"""Single entry point for the query_plan_experiment package, replacing the
empty root query_script.sh.

Usage (run from the BOLT repo root):
    python -m query_plan_experiment.cli trajectory_chain \\
        --config query_plan_experiment/configs/query_plan_smoke.yaml

    python -m query_plan_experiment.cli heldout_eval \\
        --config query_plan_experiment/configs/query_plan_smoke.yaml --arm all

    python -m query_plan_experiment.cli aggregate \\
        --config query_plan_experiment/configs/query_plan_smoke.yaml
"""

from __future__ import annotations

import fire

from .aggregate import _arms as _all_arms
from .aggregate import build_table1
from .config import load_config
from .heldout_eval import run_heldout_eval
from .trajectory_chain import run_trajectory_chain


class CLI:
    def trajectory_chain(self, config: str) -> None:
        cfg = load_config(config)
        run_trajectory_chain(cfg)

    def heldout_eval(self, config: str, arm: str) -> None:
        cfg = load_config(config)
        arms = _all_arms(cfg) if arm == "all" else [arm]
        for a in arms:
            run_heldout_eval(cfg, a)

    def aggregate(self, config: str) -> None:
        cfg = load_config(config)
        build_table1(cfg)


if __name__ == "__main__":
    fire.Fire(CLI)

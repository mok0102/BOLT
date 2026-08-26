"""Experiment config: one YAML per experiment, resolved into absolute paths.
Mirrors peptide_experiment/config.py's shape; see
imp_plan/02_query_plan_reimplementation_plan.md for the full design and for
every field's rationale.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import task_splits

BOLT_ROOT = Path(__file__).resolve().parents[1]
# optimization/query_plans/query_plan_optimization/ isn't set up as a nested
# importable package (its own internal cross-imports are flat, e.g.
# `from your_tasks.your_objective_functions import ...`, assuming this dir
# itself is on sys.path) -- add it here, once, so steps.py can import
# DatabaseObjective in-process the same way peptide's config.py adds
# optimization/peptides/ for `from apex_oracle import apex_wrapper`.
sys.path.insert(0, str(BOLT_ROOT / "optimization" / "query_plans" / "query_plan_optimization"))


@dataclass
class ExperimentConfig:
    experiment_id: str
    milestones: list[int] = field(default_factory=lambda: [50, 893, 1138, 1426])
    oracle_budget: int = 4000
    # Paper value (p.5-6). Optimize's own num_initialization_points default
    # (200) is a generic LOL-BO-template leftover -- always pass this
    # explicitly, never rely on that default.
    init_size: int = 50
    bsz: int = 1
    sft_epochs: int = 1
    torchtune_config: str = "qwen_2_5_3B_lora.yaml"
    torchtune_recipe: str = "lora_finetune_distributed"
    base_checkpoint_dir: Path = field(
        default_factory=lambda: BOLT_ROOT / "fine-tuning" / "query_plans" / "ckpt" / "Qwen2.5-3B-Instruct"
    )
    max_train_tasks: int | None = None  # None -> max(milestones)
    cuda_visible_devices: str | None = None
    # Smoke-test escape hatch, same idiom as peptide's: override the real
    # 99-task held-out set with a tiny list of workload names.
    heldout_tasks_override: list[str] | None = None
    which_query_language: str = "aliases"
    allow_cross_joins: bool = True
    # False (default): cold-start the VAE with random weights every run_bo()
    # call. The paper's real VAE (appendix C.1) is a pre-trained, never-
    # retrained-during-BO artifact from external prior work (Tao et al.
    # 2025) that isn't available anywhere in this repo (only a placeholder
    # checkpoint file, never committed even via git-lfs) -- confirmed with
    # the user this is an acceptable, explicitly-documented fidelity
    # deviation for this pass, mirroring peptide's own use_pretrained_vae
    # knob. Set True + vae_statedict_path once a real checkpoint is sourced.
    use_pretrained_vae: bool = False
    vae_statedict_path: str | None = None
    # Fixed per-query timeout (seconds) used only for this package's own
    # in-process init-candidate scoring (steps.py::sample_and_build_init) --
    # matches DatabaseObjective's own constructor default. The actual BO
    # subprocess's oracle-call timeout is governed independently by the
    # unmodified LOLBO script's own adaptive timeout_strategy machinery, not
    # this value.
    query_timeout_secs: float = 100.0
    # Top-N feasible-by-construction (no similarity-style constraint exists
    # for this domain, see make_train_data_csv.py) trajectory candidates
    # pooled per training task for the SFT dataset. Reuses init_size's scale
    # (both trace back to the paper's "50 plans" convention) rather than
    # inventing an unrelated separate number.
    sft_top_n_per_task: int = 50
    # Oracle-call checkpoints for aggregate.py's Table-1-style output. None
    # (default): report only the terminal oracle_budget (no intermediate k
    # confirmed by the paper facts this stage targets, unlike peptide's
    # Table 11 k=[1,100,200,500,1000]) -- override with an explicit list for
    # a scaling-curve-style breakdown if wanted later.
    table_k_checkpoints: list[int] | None = None
    # Interpreter override for optimization/query_plans/ subprocess calls,
    # which ships its own poetry-managed dependency set (lightning, botorch,
    # gpytorch, psycopg, sqlglot, pydot), distinct from what
    # fine-tuning/query_plans needs (torchtune/transformers/fire). None
    # (default): use sys.executable, i.e. assume one shared venv covers both
    # -- verify this by hand first (see the plan's verification-plan step
    # 1c) rather than trusting this default blindly.
    lolbo_python: str | None = None

    bolt_root: Path = BOLT_ROOT
    heldout_tasks: list[str] = field(default_factory=task_splits.heldout_workloads)

    def __post_init__(self) -> None:
        self.base_checkpoint_dir = Path(self.base_checkpoint_dir)
        if not self.base_checkpoint_dir.is_absolute():
            self.base_checkpoint_dir = self.bolt_root / self.base_checkpoint_dir
        self.milestones = sorted(self.milestones)
        if self.heldout_tasks_override is not None:
            self.heldout_tasks = list(self.heldout_tasks_override)

    # -- derived paths --------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self.bolt_root / "runs" / self.experiment_id

    @property
    def trajectories_dir(self) -> Path:
        return self.run_dir / "trajectories"

    @property
    def trajectories_csv_dir(self) -> Path:
        return self.run_dir / "trajectories_csv"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def milestones_dir(self) -> Path:
        return self.run_dir / "milestones"

    @property
    def aggregate_dir(self) -> Path:
        return self.run_dir / "aggregate"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    @property
    def heldout_dir(self) -> Path:
        return self.run_dir / "heldout"

    def train_task_workloads(self) -> list[str]:
        n = self.max_train_tasks if self.max_train_tasks is not None else max(self.milestones)
        return task_splits.train_workloads()[:n]

    def milestone_checkpoint_dir(self, milestone: int) -> Path:
        return self.checkpoints_dir / f"BOLT-{milestone}" / f"epoch_{self.sft_epochs - 1}"

    def ensure_dirs(self) -> None:
        for d in (
            self.trajectories_dir,
            self.trajectories_csv_dir,
            self.checkpoints_dir,
            self.milestones_dir,
            self.aggregate_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_absolute():
        path = BOLT_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)
    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    return cfg

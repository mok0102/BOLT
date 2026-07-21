"""Experiment config: one YAML per experiment, resolved into absolute paths.

See imp_plan/01_peptide_reimplementation_plan.md for why this replaces the
hardcoded shell variables the old root scripts used.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BOLT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOLT_ROOT / "optimization" / "peptides"))
from apex_oracle.task_splits import HELDOUT20_TASKS, HELDOUT100_TASKS  # noqa: E402


@dataclass
class ExperimentConfig:
    experiment_id: str
    milestones: list[int]
    oracle_budget: int
    init_size: int
    bsz: int
    sft_epochs: int
    similarity_threshold: float = 0.75
    task_specific_args: str = "bacteria_0"
    torchtune_config: str = "qwen_2_5_3B_lora.yaml"
    torchtune_recipe: str = "lora_finetune_distributed"
    base_checkpoint_dir: Path = field(
        default_factory=lambda: BOLT_ROOT / "fine-tuning" / "peptides" / "ckpt" / "Qwen2.5-3B-Instruct"
    )
    max_train_tasks: int | None = None  # None -> max(milestones)
    # e.g. "5" to pin all subprocesses to one specific GPU (matches the
    # CUDA_VISIBLE_DEVICES=N pattern the old root scripts used). None -> don't set it.
    cuda_visible_devices: str | None = None
    # Smoke-test escape hatch: override both held-out sets with a tiny list of
    # task indices, so a smoke run doesn't try to sweep the real 20/100-task
    # paper held-out sets. None -> use the real paper splits.
    heldout_tasks_override: list[int] | None = None
    # Oracle-call checkpoints for Table 11 / Figure 1-2 style aggregation.
    # Paper default is [1, 100, 200, 500, 1000] (Table 11); override to
    # something smaller than oracle_budget for a smoke run.
    table_k_checkpoints: list[int] = field(default_factory=lambda: [1, 100, 200, 500, 1000])

    # ORPT (Stage 4): when True, an ORPT-<m> DPO stage is trained on top of
    # every BOLT-<m>, and -- critically -- tasks sampled after the first
    # milestone come from the latest ORPT-<m> checkpoint rather than BOLT-<m>
    # (see trajectory_chain.py's checkpoint_to_sample_from). False by default
    # so existing BOLT-only configs are unaffected.
    build_orpt: bool = False
    orpt_epochs: int = 1
    orpt_pairs_per_task: int = 1000  # mirrors make_dpo_train_data_csv.py's own default
    orpt_beta: float = 0.1  # DPOLoss's beta (reference-relative logit scale)
    orpt_lr: float = 3e-4  # DPO optimizer learning rate
    # "feasible_only" (default, original behavior): preference pairs are built
    # only from similarity-constraint-feasible candidates, ranked by objective
    # score alone. "lexicographic": infeasible candidates are kept and ranked
    # *behind* any feasible candidate regardless of score -- the fix for
    # naive DPO proposing constraint-violating sequences at generation time
    # (see experiments/constraint_violation/ and make_dpo_train_data_csv.py's
    # sample_pairs()/_pick_chosen_rejected() docstrings for the full rationale).
    orpt_pairing_mode: str = "feasible_only"
    orpt_torchtune_config: str = "qwen_2_5_3B_lora_dpo.yaml"
    orpt_torchtune_recipe: str = "lora_dpo_distributed"

    bolt_root: Path = BOLT_ROOT
    heldout20_tasks: list[int] = field(default_factory=lambda: list(HELDOUT20_TASKS))
    heldout100_tasks: list[int] = field(default_factory=lambda: list(HELDOUT100_TASKS))

    def __post_init__(self) -> None:
        self.base_checkpoint_dir = Path(self.base_checkpoint_dir)
        if not self.base_checkpoint_dir.is_absolute():
            self.base_checkpoint_dir = self.bolt_root / self.base_checkpoint_dir
        self.milestones = sorted(self.milestones)
        if self.heldout_tasks_override is not None:
            self.heldout20_tasks = list(self.heldout_tasks_override)
            self.heldout100_tasks = list(self.heldout_tasks_override)

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
    def orpt_pairs_dir(self) -> Path:
        return self.run_dir / "orpt_pairs"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    def heldout_dir(self, task_set: str) -> Path:
        assert task_set in ("heldout20", "heldout100")
        return self.run_dir / task_set

    def heldout_tasks(self, task_set: str) -> list[int]:
        assert task_set in ("heldout20", "heldout100")
        return self.heldout20_tasks if task_set == "heldout20" else self.heldout100_tasks

    def train_task_range(self) -> range:
        n = self.max_train_tasks if self.max_train_tasks is not None else max(self.milestones)
        return range(0, n)

    def milestone_checkpoint_dir(self, milestone: int) -> Path:
        return self.checkpoints_dir / f"BOLT-{milestone}" / f"epoch_{self.sft_epochs - 1}"

    def orpt_checkpoint_dir(self, milestone: int) -> Path:
        """ORPT-<milestone>'s final checkpoint: a DPO stage trained on top of
        that same milestone's own BOLT-<milestone>.
        """
        return self.checkpoints_dir / f"ORPT-{milestone}" / f"epoch_{self.orpt_epochs - 1}"

    def ensure_dirs(self) -> None:
        for d in (
            self.trajectories_dir,
            self.trajectories_csv_dir,
            self.checkpoints_dir,
            self.milestones_dir,
            self.aggregate_dir,
            self.orpt_pairs_dir,
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
        # Also pins any in-process CUDA use (e.g. apex_wrapper called directly
        # from steps.py), not just the subprocesses steps.py launches.
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    return cfg

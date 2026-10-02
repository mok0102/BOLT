"""YAML configuration for the Branin BOLT/MI-ORPT experiment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .task_splits import TaskManifest, TaskRecord, load_manifest

BOLT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    experiment_id: str
    # Required: path to a frozen task manifest (synthetic_experiment/
    # task_splits.py::build_manifest/save_manifest) -- replaces the old
    # per-config task_seed + train_task_t_override/heldout_task_t_override
    # scalar-t sampling, which regenerated tasks from a seed on every load
    # and was never persisted.
    task_manifest: Path
    milestones: list[int] = field(default_factory=lambda: [10, 20])
    oracle_budget: int = 50
    init_size: int = 10
    bsz: int = 1
    sft_epochs: int = 1
    num_train_tasks: int = 100
    num_heldout_tasks: int = 20
    bo_seed: int = 42
    bo_candidate_pool_size: int = 2048
    bo_lengthscale: float = 0.2
    bo_ucb_beta: float = 2.0
    trajectory_parallel_workers: int = 2
    # random makes a dependency-light end-to-end baseline/smoke run. llm
    # samples coordinates from the latest BOLT/ORPT checkpoint.
    proposal_source: str = "random"
    sampling_temperature: float = 0.7
    sft_top_n_per_task: int = 20
    torchtune_config: str = "qwen_2_5_3B_lora.yaml"
    torchtune_recipe: str = "lora_finetune_distributed"
    base_checkpoint_dir: Path = field(
        default_factory=lambda: BOLT_ROOT / "fine-tuning/query_plans/ckpt/Qwen2.5-3B-Instruct"
    )
    cuda_visible_devices: str | None = None
    build_orpt: bool = False
    build_stbo: bool = False
    build_pogpe: bool = False
    build_sgpe: bool = False
    build_mtbo: bool = False
    mtbo_top_n_per_task: int = 20
    mtbo_max_train_points: int = 256
    build_optformer: bool = False
    build_llambo: bool = False
    gp_expert_top_n_per_task: int = 30
    optformer_epochs: int = 1
    optformer_context_length: int = 20
    optformer_windows_per_task: int = 50
    llambo_candidates: int = 8
    llambo_mc_samples: int = 4
    llambo_context_length: int = 20
    llambo_max_input_tokens: int = 1_000_000
    llambo_alpha: float = 0.1
    orpt_epochs: int = 1
    orpt_beta: float = 0.1
    orpt_lr: float = 2e-5
    orpt_torchtune_config: str = "qwen_2_5_3B_lora_dpo.yaml"
    orpt_torchtune_recipe: str = "lora_dpo_distributed"
    mi_num_backgrounds: int = 8
    mi_target_pairs_per_task: int = 1
    mi_max_candidates_per_task: int = 3
    mi_tau_q: float = 1.0
    mi_z_min: float = 1.96
    mi_delta_t: float = 0.0
    mi_seed: int = 42
    mi_parallel_workers: int = 8
    mi_bo_steps: int = 1
    # MI one-step BO can use a greedier, more local acquisition model than
    # the regular trajectory/evaluation BO without changing that baseline.
    mi_bo_candidate_pool_size: int | None = None
    mi_bo_lengthscale: float = 0.2
    mi_bo_ucb_beta: float = 2.0
    # candidate_acquired_best counts both the intervention and the point
    # subsequently acquired because of it, without letting the shared
    # background incumbent mask candidate differences.
    mi_utility_mode: str = "acquired_score"
    table_k_checkpoints: list[int] | None = None
    bolt_root: Path = BOLT_ROOT

    def __post_init__(self) -> None:
        self.milestones = sorted(set(self.milestones))
        if not self.milestones or self.milestones[0] < 1 or self.milestones[-1] > self.num_train_tasks:
            raise ValueError("milestones must be within the training task count")
        if self.init_size < 2 or self.oracle_budget < 1 or self.bsz < 1:
            raise ValueError("init_size >= 2, oracle_budget >= 1, and bsz >= 1 are required")
        if self.trajectory_parallel_workers < 1:
            raise ValueError("trajectory_parallel_workers must be at least 1")
        if self.bo_candidate_pool_size < 1 or self.bo_lengthscale <= 0 or self.bo_ucb_beta < 0:
            raise ValueError("invalid regular BO acquisition settings")
        if self.proposal_source not in {"random", "llm"}:
            raise ValueError("proposal_source must be random or llm")
        if self.build_orpt and (self.mi_num_backgrounds < 2 or self.mi_max_candidates_per_task < 2):
            raise ValueError("MI-ORPT needs at least two backgrounds and two candidates")
        if self.mi_tau_q <= 0 or self.mi_target_pairs_per_task < 1:
            raise ValueError("invalid MI distribution/pair settings")
        if self.mi_parallel_workers < 1:
            raise ValueError("mi_parallel_workers must be at least 1")
        if min(self.mtbo_top_n_per_task, self.mtbo_max_train_points,
               self.gp_expert_top_n_per_task, self.optformer_epochs, self.optformer_context_length,
               self.optformer_windows_per_task, self.llambo_candidates, self.llambo_mc_samples,
               self.llambo_context_length, self.llambo_max_input_tokens) < 1:
            raise ValueError("baseline sizes, epochs, and token budget must be positive")
        if self.llambo_alpha < 0:
            raise ValueError("llambo_alpha must be nonnegative")
        if self.mi_bo_steps < 0:
            raise ValueError("mi_bo_steps must be nonnegative")
        if (
            (self.mi_bo_candidate_pool_size is not None and self.mi_bo_candidate_pool_size < 1)
            or self.mi_bo_lengthscale <= 0
            or self.mi_bo_ucb_beta < 0
        ):
            raise ValueError("invalid MI one-step BO acquisition settings")
        if self.mi_utility_mode not in {
            "acquired_score", "candidate_acquired_best", "terminal_best", "improvement"
        }:
            raise ValueError(
                "mi_utility_mode must be acquired_score, candidate_acquired_best, "
                "terminal_best, or improvement"
            )
        self.base_checkpoint_dir = Path(self.base_checkpoint_dir)
        if not self.base_checkpoint_dir.is_absolute():
            self.base_checkpoint_dir = self.bolt_root / self.base_checkpoint_dir

        self.task_manifest = Path(self.task_manifest)
        if not self.task_manifest.is_absolute():
            self.task_manifest = self.bolt_root / self.task_manifest
        self._manifest = load_manifest(self.task_manifest)
        if self.num_train_tasks > len(self._manifest.train):
            raise ValueError(
                f"num_train_tasks={self.num_train_tasks} exceeds the manifest's "
                f"{len(self._manifest.train)} training tasks ({self.task_manifest})"
            )
        if self.num_heldout_tasks > len(self._manifest.heldout):
            raise ValueError(
                f"num_heldout_tasks={self.num_heldout_tasks} exceeds the manifest's "
                f"{len(self._manifest.heldout)} held-out tasks ({self.task_manifest})"
            )

    @property
    def manifest(self) -> TaskManifest:
        return self._manifest

    @property
    def train_tasks(self) -> tuple[TaskRecord, ...]:
        """Stable prefix of the manifest's training split -- each split has
        its own random stream (see build_manifest), so a smaller
        num_train_tasks is always a prefix of a larger one, not a resample."""
        return self._manifest.train[: self.num_train_tasks]

    @property
    def heldout_tasks(self) -> tuple[TaskRecord, ...]:
        return self._manifest.heldout[: self.num_heldout_tasks]

    @property
    def run_dir(self) -> Path:
        return self.bolt_root / "runs" / self.experiment_id

    trajectories_dir = property(lambda self: self.run_dir / "trajectories")
    checkpoints_dir = property(lambda self: self.run_dir / "checkpoints")
    milestones_dir = property(lambda self: self.run_dir / "milestones")
    orpt_pairs_dir = property(lambda self: self.run_dir / "orpt_pairs")
    heldout_dir = property(lambda self: self.run_dir / "heldout")
    aggregate_dir = property(lambda self: self.run_dir / "aggregate")
    tensorboard_dir = property(lambda self: self.run_dir / "tensorboard")

    def milestone_checkpoint_dir(self, milestone: int) -> Path:
        # Flat -- matches where `tune run`'s own `output_dir=` points, and
        # the directory `steps.py::materialize_hf_checkpoint` flattens a
        # finished checkpoint into in place (see that function's docstring
        # for why the previous `/ f"epoch_{self.sft_epochs - 1}"` suffix
        # didn't match this torchtune install's real output layout).
        return self.checkpoints_dir / f"BOLT-{milestone}"

    def mtbo_checkpoint(self, milestone: int) -> Path:
        return self.checkpoints_dir / f"MTBO-{milestone}" / "surrogate.npz"

    def optformer_checkpoint_dir(self, milestone: int) -> Path:
        return self.checkpoints_dir / f"OptFormer-{milestone}"

    def gp_expert_dir(self, milestone: int) -> Path:
        return self.run_dir / "gp_experts" / f"milestone_{milestone}"

    def orpt_checkpoint_dir(self, milestone: int) -> Path:
        return self.checkpoints_dir / f"ORPT-{milestone}"

    def ensure_dirs(self) -> None:
        for path in (self.trajectories_dir, self.checkpoints_dir, self.milestones_dir, self.aggregate_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._check_manifest_lock()

    def _check_manifest_lock(self) -> None:
        """Every pipeline stage here skips recomputation if its output
        already exists (run_bo, checkpoint_ready, build_orpt_pairs, the
        MTBO/MI caches) -- reusing an old run_dir under a new task manifest
        would silently "finish in minutes" while actually just replaying
        stale data from whatever manifest that run_dir was last used with.
        This lock makes that a loud error instead."""
        lock_path = self.run_dir / "task_manifest.lock.json"
        if lock_path.exists():
            locked = json.loads(lock_path.read_text())
            if locked.get("token") != self.manifest.token:
                raise ValueError(
                    f"{self.run_dir} was already used with task manifest token "
                    f"{locked.get('token')!r}, but this config points at manifest "
                    f"{self.task_manifest} (token {self.manifest.token!r}) -- use a new "
                    "experiment_id/run_dir for a different manifest, don't reuse this one"
                )
            return
        if any(self.trajectories_dir.glob("*.csv")):
            raise ValueError(
                f"{self.run_dir} already has trajectory data but no task_manifest.lock.json "
                "(a run from before the manifest system existed) -- use a new experiment_id/run_dir"
            )
        lock_path.write_text(json.dumps(
            {"token": self.manifest.token, "task_manifest": str(self.task_manifest)}, indent=2,
        ))


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_absolute():
        path = BOLT_ROOT / path
    cfg = ExperimentConfig(**yaml.safe_load(path.read_text()))
    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    return cfg

"""OptFormer fine-tuning (paper's LLM-as-optimizer baseline, reimplemented
against this repo's own torchtune/Qwen infra rather than the paper's
GPT-4o-mini): trains a per-milestone OptFormer-<milestone> checkpoint on
history-conditioned windows sampled from the same cumulative trajectory data
[0, milestone) BOLT-<milestone>'s own SFT dataset uses, via fine-tuning/
peptides/make_optformer_train_data_csv.py + generate_optformer_ft_data.py.
Reuses the existing SFT torchtune recipe/config unchanged (same override
pattern trajectory_chain.py::train_milestone() uses), just pointed at
OptFormer's own dataset/output_dir/epochs.
"""

from __future__ import annotations

import sys

from .config import ExperimentConfig
from .steps import _run, cleanup_intermediate_epochs

FINE_TUNING_DIR = "fine-tuning/peptides"


def train_optformer(cfg: ExperimentConfig, milestone: int):
    ckpt_dir = cfg.checkpoints_dir / f"OptFormer-{milestone}"
    final_ckpt = cfg.optformer_checkpoint_dir(milestone)
    if final_ckpt.exists():
        print(f"[optformer milestone {milestone}] checkpoint already exists at {final_ckpt}, skipping")
        return final_ckpt

    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR
    train_csv = cfg.optformer_dir / f"train_data_{milestone}.csv"
    train_jsonl = cfg.optformer_dir / f"train_data_{milestone}.jsonl"
    bin_edges_path = cfg.optformer_dir / f"score_bin_edges_{milestone}.json"

    input_csvs = [cfg.trajectories_csv_dir / f"task_{i:04d}.csv" for i in range(milestone)]
    reference_indices = [str(i) for i in range(milestone)]

    _run(
        [
            sys.executable,
            "make_optformer_train_data_csv.py",
            "--input-csv",
            *input_csvs,
            "--reference-index",
            *reference_indices,
            "--output-csv",
            train_csv,
            "--bin-edges-output",
            bin_edges_path,
            "--similarity-threshold",
            cfg.similarity_threshold,
            "--num-bins",
            cfg.optformer_num_score_bins,
            "--context-length",
            cfg.optformer_context_length,
            "--windows-per-task",
            cfg.optformer_windows_per_task,
            "--seed",
            42,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    _run(
        [
            sys.executable,
            "generate_optformer_ft_data.py",
            "--data-path",
            train_csv,
            "--save-path",
            train_jsonl,
            "--num-bins",
            cfg.optformer_num_score_bins,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    _run(
        [
            "tune",
            "run",
            "--nnodes",
            "1",
            "--nproc_per_node",
            "1",
            cfg.torchtune_recipe,
            "--config",
            f"torchtune_config/{cfg.torchtune_config}",
            f"output_dir={ckpt_dir}",
            f"dataset.data_files={train_jsonl}",
            f"tokenizer.path={cfg.base_checkpoint_dir}/vocab.json",
            f"tokenizer.merges_file={cfg.base_checkpoint_dir}/merges.txt",
            f"checkpointer.checkpoint_dir={cfg.base_checkpoint_dir}",
            f"epochs={cfg.optformer_epochs}",
            "seed=42",
            f"metric_logger.log_dir={cfg.tensorboard_dir / f'OptFormer-{milestone}'}",
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    if not final_ckpt.exists():
        raise RuntimeError(
            f"Expected OptFormer milestone {milestone} checkpoint at {final_ckpt}, but it wasn't produced"
        )
    cleanup_intermediate_epochs(ckpt_dir, final_ckpt)
    return final_ckpt

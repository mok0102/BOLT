"""Standalone reference-model log-likelihood scoring for matched-intervention
ORPT's reference-aligned candidate distribution q_t
(imp_plan/06_orpt_matched_intervention_plan.md).

At pair-construction time for milestone m, pi_ref is plain BOLT-<m> itself
(no ORPT LoRA adapter exists yet for this milestone) -- so this loads the
exact same model/tokenizer/checkpointer components the SFT config
(cfg.torchtune_config, e.g. qwen_2_5_3B_lora.yaml) already declares,
pointed at that milestone's checkpoint_dir, and reuses
torchtune.rlhf.get_batch_log_probs the same way
fine-tuning/peptides/dpo_ii/recipe.py's _combined_forward does (forward
pass + teacher-forced log-probs, no training loop). Verify once against a
real BOLT-<m> checkpoint before trusting this at scale -- this is the one
piece of the pipeline most worth a smoke test, since torchtune's
config-instantiate path isn't otherwise exercised outside `tune run`
subprocesses anywhere else in this repo.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchtune import config as tt_config
from torchtune import rlhf
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, Message


def _load_model_and_tokenizer(torchtune_config_path: Path, checkpoint_dir: Path):
    cfg = OmegaConf.load(torchtune_config_path)
    cfg.checkpointer.checkpoint_dir = str(checkpoint_dir)
    # This scorer never calls save_checkpoint() -- output_dir only exists to
    # satisfy the checkpointer's constructor, which rejects output_dir ==
    # (or nested under) checkpoint_dir. A throwaway temp dir is correct and
    # never actually written to.
    scratch_output_dir = tempfile.mkdtemp(prefix="mi_orpt_scorer_")
    cfg.checkpointer.output_dir = scratch_output_dir
    # A trained BOLT-<m> checkpoint dir is a full merged checkpoint --
    # vocab.json/merges.txt live there too, not just at the base-model path
    # the SFT config's placeholder tokenizer.path/merges_file point at.
    cfg.tokenizer.path = str(checkpoint_dir / "vocab.json")
    cfg.tokenizer.merges_file = str(checkpoint_dir / "merges.txt")

    tokenizer = tt_config.instantiate(cfg.tokenizer)

    checkpointer = tt_config.instantiate(cfg.checkpointer)
    checkpoint_dict = checkpointer.load_checkpoint()

    with torch.device("cuda"):
        model = tt_config.instantiate(cfg.model)
    model.load_state_dict(checkpoint_dict["model"], strict=False)
    model.eval()
    return model, tokenizer


def score_sequences(
    torchtune_config_path: Path,
    checkpoint_dir: Path,
    context: str,
    sequences: list[str],
    system_prompt: str,
    batch_size: int = 32,
) -> list[float]:
    """Returns one sequence-level log pi_ref(seq | context) per input
    sequence, teacher-forced under the given checkpoint. `context` is the
    reference peptide (the "user" turn); `sequences` are the assistant
    completions to score -- the same message shape
    make_dpo_train_data_csv.py::make_messages builds."""
    model, tokenizer = _load_model_and_tokenizer(torchtune_config_path, checkpoint_dir)

    log_probs: list[float] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]
            batch_ids = []
            batch_labels = []
            for seq in batch_seqs:
                messages = [
                    Message(role="system", content=system_prompt, masked=True),
                    Message(role="user", content=context, masked=True),
                    Message(role="assistant", content=seq, masked=False),
                ]
                tokens, mask = tokenizer.tokenize_messages(messages)
                ids = torch.tensor(tokens, dtype=torch.long)
                # mask=True marks non-assistant tokens per torchtune's own
                # convention (prompt/system masked out of the loss).
                labels = ids.clone()
                labels[torch.tensor(mask, dtype=torch.bool)] = CROSS_ENTROPY_IGNORE_IDX
                batch_ids.append(ids)
                batch_labels.append(labels)

            max_len = max(t.numel() for t in batch_ids)
            padded_ids = torch.stack(
                [torch.nn.functional.pad(t, (0, max_len - t.numel()), value=tokenizer.pad_id) for t in batch_ids]
            ).cuda()
            padded_labels = torch.stack(
                [
                    torch.nn.functional.pad(t, (0, max_len - t.numel()), value=CROSS_ENTROPY_IGNORE_IDX)
                    for t in batch_labels
                ]
            ).cuda()

            logits = model(padded_ids)
            batch_log_probs = rlhf.get_batch_log_probs(logits, padded_labels)
            log_probs.extend(batch_log_probs.tolist())

    return log_probs

"""Distributed LoRA recipe for the dpo_ii ablation ("standard ORPT DPO loss,
plus an additive infeasible-singles suppression term").

Subclasses torchtune's installed LoRADPORecipeDistributed the same way
../fa_orpt/recipe.py does (that recipe's own docstring explains why: no
config-only way to thread extra per-example data into the stock recipe's
train() loop, and the `recipes` package this base class lives in intentionally
blocks plain `import recipes...`, so it's loaded via importlib instead).

Unlike fa_orpt/recipe.py, the main DPO dataset/dataloader (`self._dataloader`,
`self._loss_fn`) is left completely untouched -- inherited from the parent
as-is, same standard feasible_only preference pairs + stock
torchtune.rlhf.loss.DPOLoss. This recipe only *adds* a second, independent
dataloader of individually-sampled infeasible completions
(../make_infeasible_singles_csv.py's output) and an additive loss term
(dpo_ii/loss.py's InfeasibleSuppressionLoss) on top.

Per confirmed design: rather than running 2 extra forward passes for the
infeasible-singles stream (4 total forwards/step -- 2x today's DPO compute),
chosen + rejected + infeasible-singles are concatenated into one combined
batch and forwarded together, so there are still only 2 total forward passes
per step (1 policy, 1 reference) -- see _combined_forward(), which extends
the parent's own concatenated_forward() "concatenate then split by known
slice boundaries" pattern from 2 groups to 3.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torchtune
from omegaconf import DictConfig
from torch.distributed import init_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, rlhf, training, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, padded_collate_sft
from torchtune.modules.peft import disable_adapter
from tqdm import tqdm

log = utils.get_logger("DEBUG")

# Same ROOT computation torchtune._cli.run uses to locate built-in recipes,
# and the same load pattern ../fa_orpt/recipe.py already uses.
_TORCHTUNE_SITE_PACKAGES_ROOT = Path(torchtune.__file__).resolve().parent.parent
_DPO_RECIPE_PATH = _TORCHTUNE_SITE_PACKAGES_ROOT / "recipes" / "lora_dpo_distributed.py"


def _load_dpo_recipe_module():
    spec = importlib.util.spec_from_file_location("_torchtune_lora_dpo_distributed", _DPO_RECIPE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load torchtune's lora_dpo_distributed recipe from {_DPO_RECIPE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dpo_recipe_module = _load_dpo_recipe_module()
LoRADPORecipeDistributed = _dpo_recipe_module.LoRADPORecipeDistributed


def _pad_seq_dim(x: torch.Tensor, target_len: int, value: int) -> torch.Tensor:
    """Pads a 2D (batch, seq) tensor's sequence dim up to target_len -- needed
    because padded_collate_dpo (main DPO batch) and padded_collate_sft
    (infeasible-singles batch) each pad only within their own batch, so the
    two groups' sequence lengths can differ before concatenation."""
    if x.shape[1] == target_len:
        return x
    return F.pad(x, (0, target_len - x.shape[1]), value=value)


class LoRADPOIIRecipeDistributed(LoRADPORecipeDistributed):
    """LoRADPORecipeDistributed, plus a second infeasible-singles dataloader
    and a combined-batch forward pass that adds InfeasibleSuppressionLoss's
    mean on top of the untouched stock DPOLoss."""

    def setup(self, cfg: DictConfig) -> None:
        super().setup(cfg=cfg)
        self._ii_loss_fn = config.instantiate(cfg.ii_loss)
        utils.log_rank_zero(log, "dpo_ii loss is initialized.")
        self._ii_sampler, self._ii_dataloader = self._setup_ii_data(
            cfg_dataset=cfg.ii_dataset,
            batch_size=cfg.get("ii_batch_size", cfg.batch_size),
        )
        self._ii_iter = iter(self._ii_dataloader)
        utils.log_rank_zero(log, "dpo_ii infeasible-singles dataset and sampler are initialized.")

    def _setup_ii_data(
        self,
        cfg_dataset: DictConfig,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
        sampler = DistributedSampler(ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=0)
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=partial(
                padded_collate_sft,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
            ),
        )
        return sampler, dataloader

    def _next_ii_batch(self) -> dict:
        """The infeasible-singles dataloader's epoch length generally differs
        from the main DPO dataloader's, so it's cycled independently rather
        than iterated in lockstep."""
        try:
            return next(self._ii_iter)
        except StopIteration:
            self._ii_iter = iter(self._ii_dataloader)
            return next(self._ii_iter)

    def _combined_forward(
        self,
        model: torch.nn.Module,
        dpo_batch: Tuple[torch.Tensor, torch.Tensor],
        ii_batch: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenates chosen+rejected (dpo_batch) and infeasible-singles
        (ii_batch) into one batch, runs a single forward pass, and splits the
        resulting per-example log probs back into (chosen, rejected,
        infeasible) by known slice boundaries -- extends the parent's
        concatenated_forward()'s own "concatenate then split" pattern from 2
        groups to 3."""
        dpo_ids, dpo_labels = dpo_batch
        dpo_ids = dpo_ids.to(self._device)
        dpo_labels = dpo_labels.to(self._device)
        ii_ids = ii_batch["tokens"].to(self._device)
        ii_labels = ii_batch["labels"].to(self._device)

        max_len = max(dpo_ids.shape[1], ii_ids.shape[1])
        dpo_ids = _pad_seq_dim(dpo_ids, max_len, self._tokenizer.pad_id)
        dpo_labels = _pad_seq_dim(dpo_labels, max_len, CROSS_ENTROPY_IGNORE_IDX)
        ii_ids = _pad_seq_dim(ii_ids, max_len, self._tokenizer.pad_id)
        ii_labels = _pad_seq_dim(ii_labels, max_len, CROSS_ENTROPY_IGNORE_IDX)

        combined_ids = torch.cat([dpo_ids, ii_ids], dim=0)
        combined_labels = torch.cat([dpo_labels, ii_labels], dim=0)

        with self.activations_handling_ctx:
            all_logits = model(combined_ids)
        all_log_probs = rlhf.get_batch_log_probs(all_logits, combined_labels)
        del all_logits

        n_pair = dpo_ids.shape[0]
        len_chosen = n_pair // 2
        chosen_log_probs = all_log_probs[:len_chosen]
        rejected_log_probs = all_log_probs[len_chosen:n_pair]
        ii_log_probs = all_log_probs[n_pair:]
        return chosen_log_probs, rejected_log_probs, ii_log_probs

    def train(self) -> None:
        """Copy of the parent's train() loop. The only deltas: an
        infeasible-singles batch is pulled each step and folded into a
        combined forward pass (_combined_forward instead of
        concatenated_forward), and the final loss is
        dpo_loss.mean() + ii_loss.mean() instead of just dpo_loss.mean()."""
        training.cleanup_before_training()
        self._optimizer.zero_grad()

        t0 = time.perf_counter()

        running_loss = 0
        running_metrics = {
            "rewards/chosen": 0,
            "rewards/rejected": 0,
            "rewards/accuracies": 0,
            "log_probs/chosen": 0,
            "log_probs/rejected": 0,
            "dpo_ii/ii_loss": 0,
            "dpo_ii/ii_mean_u": 0,
        }
        num_tokens = 0

        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch, disable=not (self.rank == 0))
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps) == self.max_steps_per_epoch
                ):
                    break

                num_tokens += torch.tensor(batch[0].numel())

                ii_batch = self._next_ii_batch()

                policy_chosen_log_probs, policy_rejected_log_probs, policy_ii_log_probs = self._combined_forward(
                    self._model, batch, ii_batch
                )

                with torch.no_grad(), disable_adapter(self._model):
                    reference_chosen_log_probs, reference_rejected_log_probs, reference_ii_log_probs = (
                        self._combined_forward(self._model, batch, ii_batch)
                    )

                dpo_loss, chosen_rewards, rejected_rewards = self._loss_fn(
                    policy_chosen_log_probs,
                    policy_rejected_log_probs,
                    reference_chosen_log_probs,
                    reference_rejected_log_probs,
                )
                ii_per_example_loss = self._ii_loss_fn(policy_ii_log_probs, reference_ii_log_probs)
                ii_loss = ii_per_example_loss.mean()

                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                loss = (dpo_loss.mean() + ii_loss) / self._gradient_accumulation_steps

                scaling_factor = 1 / self._gradient_accumulation_steps
                running_loss += loss
                running_metrics["rewards/chosen"] += scaling_factor * chosen_rewards.mean()
                running_metrics["rewards/rejected"] += scaling_factor * rejected_rewards.mean()
                running_metrics["rewards/accuracies"] += scaling_factor * reward_accuracies.mean()
                running_metrics["log_probs/chosen"] += scaling_factor * policy_chosen_log_probs.detach().mean()
                running_metrics["log_probs/rejected"] += scaling_factor * policy_rejected_log_probs.detach().mean()
                running_metrics["dpo_ii/ii_loss"] += scaling_factor * ii_loss.detach()
                running_metrics["dpo_ii/ii_mean_u"] += scaling_factor * (
                    policy_ii_log_probs.detach() - reference_ii_log_probs.detach()
                ).mean()

                loss.backward()

                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    torch.distributed.all_reduce(running_loss)
                    torch.distributed.all_reduce(num_tokens)

                    for key in running_metrics:
                        torch.distributed.all_reduce(running_metrics[key], op=torch.distributed.ReduceOp.AVG)

                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()

                    self.global_step += 1

                    loss_to_log = running_loss.item()
                    pbar.update(1)
                    pbar.set_description(f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}")

                    if self.global_step % self._log_every_n_steps == 0 and self._is_rank_zero:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens / (time_per_step * self.world_size),
                            "rewards/chosen": running_metrics["rewards/chosen"].cpu(),
                            "rewards/rejected": running_metrics["rewards/rejected"].cpu(),
                            "rewards/accuracies": running_metrics["rewards/accuracies"].cpu(),
                            "rewards/margins": (
                                running_metrics["rewards/chosen"] - running_metrics["rewards/rejected"]
                            ).cpu(),
                            "log_probs/chosen": running_metrics["log_probs/chosen"].cpu(),
                            "log_probs/rejected": running_metrics["log_probs/rejected"].cpu(),
                            "dpo_ii/ii_loss": running_metrics["dpo_ii/ii_loss"].cpu(),
                            "dpo_ii/ii_mean_u": running_metrics["dpo_ii/ii_mean_u"].cpu(),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(training.get_memory_stats(device=self._device))
                        self._metric_logger.log_dict(log_dict, step=self.global_step)

                    running_loss = 0
                    running_metrics = {key: 0 for key in running_metrics}
                    num_tokens = 0

                    t0 = time.perf_counter()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for `tune run dpo_ii/recipe.py --config ...` -- see
    peptide_experiment/orpt.py::train_orpt_milestone() for the invocation.
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    init_process_group("cuda:nccl,cpu:gloo")
    if cfg.get("fsdp_cpu_offload", False):
        training.set_torch_num_threads()

    config.log_config(recipe_name="LoRADPOIIRecipeDistributed", cfg=cfg)

    recipe = LoRADPOIIRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

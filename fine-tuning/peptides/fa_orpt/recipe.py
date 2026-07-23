"""Distributed LoRA recipe for the fa_orpt loss.

torchtune's own `lora_dpo_distributed` recipe (installed at
<torchtune_root>/recipes/lora_dpo_distributed.py -- not part of this repo)
hardcodes its `train()` loop to unpack a plain (input_ids, labels) 2-tuple
per batch and call `self._loss_fn(policy_chosen_logps, policy_rejected_logps,
reference_chosen_logps, reference_rejected_logps)` with exactly those 4
tensors. There's no config-only way to thread per-example feasibility labels
through that call, since torchtune.datasets.PreferenceDataset /
torchtune.data.padded_collate_dpo don't carry any extra per-example fields
either. So this recipe subclasses the installed LoRADPORecipeDistributed and
overrides only `_setup_data` (to use ../fa_orpt/dataset.py's
feasibility-aware collate) and `train()` (to unpack the extra feasibility
tensors and pass them into the loss, plus log fa_orpt-specific diagnostics) --
model setup, optimizer setup, checkpointing, and `concatenated_forward`
(the actual policy/reference log-prob computation) are all inherited
unmodified.

The `recipes` package this recipe's base class lives in intentionally raises
on `import recipes...` (see its `__init__.py`) to keep pytest from picking it
up, so we load the recipe file directly via importlib (the same
spec_from_file_location pattern ../make_dpo_train_data_csv.py's
load_reference_sequence() already uses for refseqs.py) instead of a plain
import statement.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple

import torch
import torchtune
from omegaconf import DictConfig, ListConfig
from torch.distributed import init_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, training, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import disable_adapter
from tqdm import tqdm

from fa_orpt.dataset import padded_collate_fa_orpt

log = utils.get_logger("DEBUG")

# Same ROOT computation torchtune._cli.run uses to locate built-in recipes.
_TORCHTUNE_SITE_PACKAGES_ROOT = Path(torchtune.__file__).resolve().parent.parent
_DPO_RECIPE_PATH = _TORCHTUNE_SITE_PACKAGES_ROOT / "recipes" / "lora_dpo_distributed.py"


def _load_dpo_recipe_module():
    spec = importlib.util.spec_from_file_location(
        "_torchtune_lora_dpo_distributed", _DPO_RECIPE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load torchtune's lora_dpo_distributed recipe from {_DPO_RECIPE_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dpo_recipe_module = _load_dpo_recipe_module()
LoRADPORecipeDistributed = _dpo_recipe_module.LoRADPORecipeDistributed


class LoRAFAORPTRecipeDistributed(LoRADPORecipeDistributed):
    """LoRADPORecipeDistributed, with a feasibility-aware dataloader and a
    train() loop that threads per-example feasibility labels into the loss.
    """

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """Identical to the parent's _setup_data, except for the collate_fn
        (padded_collate_fa_orpt instead of padded_collate_dpo)."""
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = DistributedSampler(
            ds, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle, seed=0
        )

        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=partial(
                padded_collate_fa_orpt,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
            ),
        )

        utils.log_rank_zero(log, "Dataset and Sampler are initialized.")

        return sampler, dataloader

    def train(self) -> None:
        """Copy of the parent's train() loop. The only deltas: batch is a
        4-tuple (input_ids, labels, chosen_feasible, rejected_feasible)
        instead of a 2-tuple; the extra two tensors are passed into
        self._loss_fn(); and fa_orpt/* diagnostics are computed from the
        (already detached) chosen_rewards/rejected_rewards == u_plus/u_minus
        and logged alongside the existing metrics -- no extra forward pass
        needed. concatenated_forward() (log-prob computation, both policy
        and reference) is reused completely unmodified.
        """
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
            "logits/chosen": 0,
            "logits/rejected": 0,
            # Candidate-role view (pooled across pair types, by feasibility
            # label + win/loss role instead of by pair type) -- denser
            # per-step sample counts than a pair-type breakdown would give
            # (e.g. "infeasible" pools fi's rejected side with both of ii's
            # sides), and reads directly as "what is u doing for a
            # chosen-and-feasible candidate / a feasible candidate that
            # still lost / any infeasible candidate".
            "fa_orpt/chosen_feasible_count": 0,
            "fa_orpt/chosen_feasible_mean_u": 0,
            "fa_orpt/chosen_feasible_frac_u_neg": 0,
            "fa_orpt/losing_feasible_count": 0,
            "fa_orpt/losing_feasible_mean_u": 0,
            "fa_orpt/losing_feasible_frac_u_below_keep": 0,
            "fa_orpt/infeasible_count": 0,
            "fa_orpt/infeasible_mean_u": 0,
            "fa_orpt/infeasible_frac_u_pos": 0,
            # Pair-wise margins (u_winner - u_loser within one pair) --
            # NOT recoverable from the candidate-role means above (mean of
            # differences != difference of means, since chosen_feasible/
            # infeasible pool different pair types together). Directly
            # diagnoses the DPO-style loss's original failure mode: chosen
            # and rejected both decreasing with an insignificant margin
            # between them (see module docstring).
            "fa_orpt/ff_mean_margin": 0,
            "fa_orpt/fi_count": 0,
            "fa_orpt/fi_mean_margin": 0,
        }
        num_tokens = 0

        gamma_keep = self._loss_fn.gamma_keep

        def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            count = mask.sum().clamp(min=1)
            return (x * mask.float()).sum() / count

        def pooled_mean(values_and_masks: list) -> torch.Tensor:
            """Like masked_mean, but pools (value, mask) pairs drawn from
            different tensors (e.g. the chosen slot and the rejected slot)
            into a single weighted average."""
            total = sum((v * m.float()).sum() for v, m in values_and_masks)
            count = sum(m.sum() for v, m in values_and_masks).clamp(min=1)
            return total / count

        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch, disable=not (self.rank == 0))
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                input_ids, labels, chosen_feasible, rejected_feasible = batch
                chosen_feasible = chosen_feasible.to(self._device)
                rejected_feasible = rejected_feasible.to(self._device)
                dpo_batch = (input_ids, labels)

                num_tokens += torch.tensor(dpo_batch[0].numel())

                (
                    policy_chosen_log_probs,
                    policy_rejected_log_probs,
                    policy_chosen_logits,
                    policy_rejected_logits,
                ) = self.concatenated_forward(self._model, dpo_batch)

                policy_chosen_logits_mean = policy_chosen_logits.detach().mean()
                policy_rejected_logits_mean = policy_rejected_logits.detach().mean()

                del policy_chosen_logits, policy_rejected_logits

                with torch.no_grad(), disable_adapter(self._model):
                    (
                        reference_chosen_log_probs,
                        reference_rejected_log_probs,
                        _,
                        _,
                    ) = self.concatenated_forward(self._model, dpo_batch)
                loss, chosen_rewards, rejected_rewards = self._loss_fn(
                    policy_chosen_log_probs,
                    policy_rejected_log_probs,
                    reference_chosen_log_probs,
                    reference_rejected_log_probs,
                    chosen_feasible,
                    rejected_feasible,
                )
                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                loss = loss.mean()

                loss = loss / self._gradient_accumulation_steps

                # fa_orpt diagnostics -- chosen_rewards/rejected_rewards are
                # already u_plus/u_minus, detached (see fa_orpt/loss.py).
                mask_chosen_feasible = chosen_feasible.bool()
                mask_rejected_feasible = rejected_feasible.bool()
                mask_ff = mask_chosen_feasible & mask_rejected_feasible
                mask_fi = mask_chosen_feasible & ~mask_rejected_feasible

                scaling_factor = 1 / self._gradient_accumulation_steps

                running_loss += loss
                running_metrics["rewards/chosen"] += (
                    scaling_factor * chosen_rewards.mean()
                )
                running_metrics["rewards/rejected"] += (
                    scaling_factor * rejected_rewards.mean()
                )
                running_metrics["rewards/accuracies"] += (
                    scaling_factor * reward_accuracies.mean()
                )
                running_metrics["log_probs/chosen"] += (
                    scaling_factor * policy_chosen_log_probs.detach().mean()
                )
                running_metrics["log_probs/rejected"] += (
                    scaling_factor * policy_rejected_log_probs.detach().mean()
                )
                running_metrics["logits/chosen"] += (
                    scaling_factor * policy_chosen_logits_mean
                )
                running_metrics["logits/rejected"] += (
                    scaling_factor * policy_rejected_logits_mean
                )
                mask_chosen_infeasible = ~mask_chosen_feasible
                mask_rejected_infeasible = ~mask_rejected_feasible
                running_metrics["fa_orpt/chosen_feasible_count"] += (
                    scaling_factor * mask_chosen_feasible.sum()
                )
                running_metrics["fa_orpt/chosen_feasible_mean_u"] += scaling_factor * masked_mean(
                    chosen_rewards, mask_chosen_feasible
                )
                running_metrics["fa_orpt/chosen_feasible_frac_u_neg"] += scaling_factor * masked_mean(
                    (chosen_rewards < 0).float(), mask_chosen_feasible
                )
                running_metrics["fa_orpt/losing_feasible_count"] += (
                    scaling_factor * mask_rejected_feasible.sum()
                )
                running_metrics["fa_orpt/losing_feasible_mean_u"] += scaling_factor * masked_mean(
                    rejected_rewards, mask_rejected_feasible
                )
                running_metrics["fa_orpt/losing_feasible_frac_u_below_keep"] += scaling_factor * masked_mean(
                    (rejected_rewards < gamma_keep).float(), mask_rejected_feasible
                )
                running_metrics["fa_orpt/infeasible_count"] += scaling_factor * (
                    mask_chosen_infeasible.sum() + mask_rejected_infeasible.sum()
                )
                running_metrics["fa_orpt/infeasible_mean_u"] += scaling_factor * pooled_mean(
                    [
                        (chosen_rewards, mask_chosen_infeasible),
                        (rejected_rewards, mask_rejected_infeasible),
                    ]
                )
                running_metrics["fa_orpt/infeasible_frac_u_pos"] += scaling_factor * pooled_mean(
                    [
                        ((chosen_rewards > 0).float(), mask_chosen_infeasible),
                        ((rejected_rewards > 0).float(), mask_rejected_infeasible),
                    ]
                )
                running_metrics["fa_orpt/ff_mean_margin"] += scaling_factor * masked_mean(
                    chosen_rewards - rejected_rewards, mask_ff
                )
                running_metrics["fa_orpt/fi_count"] += scaling_factor * mask_fi.sum()
                running_metrics["fa_orpt/fi_mean_margin"] += scaling_factor * masked_mean(
                    chosen_rewards - rejected_rewards, mask_fi
                )

                loss.backward()

                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    torch.distributed.all_reduce(running_loss)
                    torch.distributed.all_reduce(num_tokens)

                    for key in running_metrics:
                        torch.distributed.all_reduce(
                            running_metrics[key], op=torch.distributed.ReduceOp.AVG
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()

                    self.global_step += 1

                    loss_to_log = running_loss.item()
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens
                            / (time_per_step * self.world_size),
                            "rewards/chosen": running_metrics["rewards/chosen"].cpu(),
                            "rewards/rejected": running_metrics[
                                "rewards/rejected"
                            ].cpu(),
                            "rewards/accuracies": running_metrics[
                                "rewards/accuracies"
                            ].cpu(),
                            "rewards/margins": (
                                running_metrics["rewards/chosen"]
                                - running_metrics["rewards/rejected"]
                            ).cpu(),
                            "log_probs/chosen": running_metrics[
                                "log_probs/chosen"
                            ].cpu(),
                            "log_probs/rejected": running_metrics[
                                "log_probs/rejected"
                            ].cpu(),
                            "logits/chosen": running_metrics["logits/chosen"].cpu(),
                            "logits/rejected": running_metrics["logits/rejected"].cpu(),
                        }
                        for key in (
                            "fa_orpt/chosen_feasible_count",
                            "fa_orpt/chosen_feasible_mean_u",
                            "fa_orpt/chosen_feasible_frac_u_neg",
                            "fa_orpt/losing_feasible_count",
                            "fa_orpt/losing_feasible_mean_u",
                            "fa_orpt/losing_feasible_frac_u_below_keep",
                            "fa_orpt/infeasible_count",
                            "fa_orpt/infeasible_mean_u",
                            "fa_orpt/infeasible_frac_u_pos",
                            "fa_orpt/ff_mean_margin",
                            "fa_orpt/fi_count",
                            "fa_orpt/fi_mean_margin",
                        ):
                            log_dict[key] = running_metrics[key].cpu()
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    running_loss = 0
                    running_metrics = {key: 0 for key in running_metrics}
                    num_tokens = 0

                    t0 = time.perf_counter()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for `tune run fa_orpt/recipe.py --config ...` -- see
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

    config.log_config(recipe_name="LoRAFAORPTRecipeDistributed", cfg=cfg)

    recipe = LoRAFAORPTRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

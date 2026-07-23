"""Feasibility-aware preference dataset + collate for the fa_orpt loss.

torchtune's stock `torchtune.datasets.PreferenceDataset._prepare_sample`
only ever returns {chosen_input_ids, chosen_labels, rejected_input_ids,
rejected_labels} -- it discards every other column in the source row. Our
pairs JSONL (written by ../make_dpo_train_data_csv.py's write_jsonl() for
--pairing-mode feasibility_aware) additionally carries "chosen_feasible" /
"rejected_feasible" (0/1) per row, so this module subclasses just enough of
the stock dataset + collate machinery to carry those two labels through to
the recipe (see recipe.py), without touching torchtune's own dataset/collate
code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torchtune.data import ChosenRejectedToMessages, CROSS_ENTROPY_IGNORE_IDX, padded_collate_dpo
from torchtune.datasets import PreferenceDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


class FeasibilityAwarePreferenceDataset(PreferenceDataset):
    """Same tokenization as PreferenceDataset, plus the raw
    chosen_feasible/rejected_feasible labels carried through unchanged."""

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        tokenized_dict = super()._prepare_sample(sample)
        tokenized_dict["chosen_feasible"] = int(sample["chosen_feasible"])
        tokenized_dict["rejected_feasible"] = int(sample["rejected_feasible"])
        return tokenized_dict


def feasibility_aware_preference_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> FeasibilityAwarePreferenceDataset:
    """Config-friendly builder mirroring torchtune.datasets.preference_dataset(),
    instantiating FeasibilityAwarePreferenceDataset instead of PreferenceDataset."""

    message_transform = ChosenRejectedToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    return FeasibilityAwarePreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )


def padded_collate_fa_orpt(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same padding as torchtune.data.padded_collate_dpo (concatenated
    chosen-then-rejected order preserved -- recipe.py's reused
    concatenated_forward() relies on that order to split the batch in half),
    plus the two feasibility labels stacked into (batch,) float tensors.
    """
    concatenated_input_ids, concatenated_labels = padded_collate_dpo(
        batch, padding_idx=padding_idx, ignore_idx=ignore_idx
    )
    chosen_feasible = torch.tensor([ex["chosen_feasible"] for ex in batch], dtype=torch.float32)
    rejected_feasible = torch.tensor([ex["rejected_feasible"] for ex in batch], dtype=torch.float32)
    return concatenated_input_ids, concatenated_labels, chosen_feasible, rejected_feasible

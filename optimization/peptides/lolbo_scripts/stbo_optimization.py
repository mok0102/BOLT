import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_dir)
sys.path.append(parent_dir)

import fire
import numpy as np
import torch

from apex_oracle import apex_wrapper
from apex_oracle.init_data.create_mutations import generate_unique_mutations
from apex_oracle.refseqs import REFERENCE_SEQUENCE
from lolbo_scripts.info_transformer_vae_optimization import (
    APEXConstrainedDiverseOptimization,
)


class STBOOptimization(APEXConstrainedDiverseOptimization):
    """STBO baseline: paper's "from scratch" single-task LOL-BO run "without
    prior task knowledge" (pp.5-6). Same LOLBO machinery/objective/similarity
    constraint as BOLT, but initialized with randomly-mutated sequences
    (within the 0.75 similarity threshold) instead of LLM-proposed
    candidates -- no LLM/torchtune involvement at all for this arm.
    """

    def load_train_data(self):
        task_idx = self.constraint_types[0]
        reference_seq = REFERENCE_SEQUENCE[task_idx]

        mutations = generate_unique_mutations(
            [reference_seq],
            num_mutations=self.num_initialization_points,
            max_mutation_distance=1.0 - self.constraint_thresholds[0],
        )[0]
        self.num_initialization_points = min(
            self.num_initialization_points, len(mutations)
        )
        mutations = mutations[: self.num_initialization_points]

        scores = -apex_wrapper(mutations)[:, 0]

        # Force a fresh VAE-encode of these actual mutations rather than
        # reusing a cached train-zs file computed for a different set of
        # sequences (load_train_z()'s cache is keyed only by count, not
        # content -- see imp_plan/01_peptide_reimplementation_plan.md).
        self.init_train_z = None
        self.init_train_x = mutations
        # unsqueeze to (N, 1): LOLBOState concatenates new scores with
        # dim=-2, which requires a 2D tensor (the file-based path gets this
        # for free from pd.read_csv(...).values on a single-column CSV).
        self.init_train_y = torch.from_numpy(np.asarray(scores)).float().unsqueeze(-1)
        return self


if __name__ == "__main__":
    fire.Fire(STBOOptimization)

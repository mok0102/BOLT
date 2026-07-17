import math

import fire
import pandas as pd
import torch

import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_dir)
sys.path.append(f"{parent_dir}")

from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
from lolbo_scripts.optimize import Optimize


torch.set_num_threads(1)

PATH_TO_VAE_STATE_DICT = os.path.join(
    parent_dir,
    "uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl",
)


class APEXConstrainedDiverseOptimization(Optimize):
    """
    Run LOL-ROBOT Constrained Optimization using InfoTransformerVAE
    """

    def __init__(
        self,
        similarity: float | None = None,
        template_id: int | None = None,
        path_to_vae_statedict: str = PATH_TO_VAE_STATE_DICT,
        max_string_length: int = 50,
        task_specific_args: list
        | None = None,  # list of additional args to be passed into objective funcion
        constraint_function_ids: list
        | None = None,  # list of strings identifying the black box constraint function to use
        constraint_thresholds: list
        | None = None,  # list of corresponding threshold values (floats)
        constraint_types: list
        | None = None,  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        divf_id: str = "edit_dist",
        init_data_path: str = None,
        init_scores_path: str = None,
        init_offset_helper: int = 0,
        **kwargs,
    ):
        if constraint_types is None:
            constraint_types = []
        if constraint_thresholds is None:
            constraint_thresholds = []
        if constraint_function_ids is None:
            constraint_function_ids = []
        if task_specific_args is None:
            task_specific_args = []
        self.path_to_vae_statedict = path_to_vae_statedict
        self.max_string_length = max_string_length
        self.task_specific_args = task_specific_args
        self.divf_id = divf_id
        self.template_id = template_id
        self.similarity = similarity
        # TODO: We currently are hard coding the init data path
        self.init_data_path = init_data_path
        self.init_scores_path = init_scores_path
        self.init_offset_helper = init_offset_helper

        print("task_specific_args: ", task_specific_args)
        print("constraint_function_ids: ", constraint_function_ids)
        print("constraint_thresholds: ", constraint_thresholds)
        print("constraint_types: ", constraint_types)

        self.score_version = task_specific_args[0]

        assert len(constraint_function_ids) == len(constraint_thresholds)
        assert len(constraint_thresholds) == len(constraint_types)
        self.constraint_function_ids = constraint_function_ids  # list of strings identifying the black box constraint function to use
        self.constraint_thresholds = (
            constraint_thresholds  # list of corresponding threshold values (floats)
        )
        self.constraint_types = constraint_types  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)

        super().__init__(**kwargs)

        # add args to method args dict to be logged by wandb
        self.method_args["diverseopt"] = locals()
        del self.method_args["diverseopt"]["self"]

    def initialize_objective(self):
        # initialize objective
        self.objective = ApexConstrainedDiverseObjective(
            similarity=self.similarity,
            template_id=self.template_id,
            task_id=self.task_id,
            task_specific_args=self.task_specific_args,
            path_to_vae_statedict=self.path_to_vae_statedict,
            max_string_length=self.max_string_length,
            divf_id=self.divf_id,
            constraint_function_ids=self.constraint_function_ids,  # list ids of the black box constraints to use
            constraint_thresholds=self.constraint_thresholds,  # list of corresponding threshold values (floats)
            constraint_types=self.constraint_types,  # list of correspoding type for each thresh ("min" or "max")
        )

        # if train zs have not been pre-computed for particular vae, compute them
        #   by passing initialization selfies through vae
        if self.init_train_z is None:
            self.init_train_z = self.compute_train_zs()
        self.init_train_c = self.objective.compute_constraints(self.init_train_x)

        return self

    def compute_train_zs(self, bsz=64):
        init_zs = []
        # make sure vae is in eval mode
        self.objective.vae.eval()
        n_batches = math.ceil(len(self.init_train_x) / bsz)
        for i in range(n_batches):
            xs_batch = self.init_train_x[i * bsz : (i + 1) * bsz]
            zs, _ = self.objective.vae_forward(xs_batch)
            init_zs.append(zs.detach().cpu())
        init_zs = torch.cat(init_zs, dim=0)
        return init_zs

    def load_train_data(self):
        """Load in or randomly initialize self.num_initialization_points
        total initial data points to kick-off optimization
        Must define the following:
            self.init_train_x (a list of x's)
            self.init_train_y (a tensor of scores/y's)
            self.init_train_z (a tensor of corresponding latent space points)
        """
        # import pdb; pdb.set_trace()

        if self.init_data_path is not None:
            filename_seqs = self.init_data_path
        else:
            filename_seqs = "./apex_oracle/init_data/init_seqs.csv"

        if self.init_scores_path is not None:
            filename_scores = self.init_scores_path
        else:
            filename_scores = f"./apex_oracle/init_data/{self.score_version}_scores.csv"

        print(f"Loading data from {filename_seqs} and {filename_scores}")

        # offset is how many bacteria we have already processed and skipped in the init files
        bacteria_num = self.constraint_types[0] - self.init_offset_helper

        file_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(f"{file_dir}")

        df = pd.read_csv(filename_seqs, header=None)
        train_x_seqs = df.values.squeeze().tolist()
        train_x_seqs = train_x_seqs[bacteria_num * 1000 : (bacteria_num + 1) * 1000]

        df = pd.read_csv(filename_scores, header=None)
        train_y = torch.from_numpy(df.values).float()
        train_y = train_y[bacteria_num * 1000 : (bacteria_num + 1) * 1000]

        self.num_initialization_points = min(
            self.num_initialization_points, len(train_x_seqs)
        )
        self.init_train_x = train_x_seqs[0 : self.num_initialization_points]
        train_y = train_y[0 : self.num_initialization_points]
        self.init_train_y = train_y  # .unsqueeze(-1)
        # Force a fresh VAE-encode of these actual sequences rather than
        # reusing a cached train-zs file computed for a different task's
        # init_train_x (load_train_z()'s cache is keyed only by row count and
        # a fixed path, not by content -- see stbo_optimization.py's identical
        # comment/fix, and imp_plan/01_peptide_reimplementation_plan.md).
        self.init_train_z = None
        return self


if __name__ == "__main__":
    fire.Fire(APEXConstrainedDiverseOptimization)

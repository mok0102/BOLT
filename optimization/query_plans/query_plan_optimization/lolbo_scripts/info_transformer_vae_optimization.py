import sys

sys.path.append("../")
import traceback
import fire
from lolbo_scripts.optimize import Optimize
from lolbo.info_transformer_vae_objective import InfoTransformerVAEObjective
import math
import pandas as pd
import torch
import math
import wandb
import os
import numpy as np
from lolbo_scripts.create_initialization_data import create_init_data_aliases
from utils.set_seed import set_seed

torch.set_num_threads(1)
LLM_GEN_TIMEOUT = 5  # timeout used to generate LLM init data from BOLT


class InfoTransformerVAEOptimization(Optimize):
    """
    Run LOLBO Optimization with InfoTransformerVAE
    Args:
        path_to_vae_statedict: Path to state dict of pretrained VAE,
            (without a limit we can run into OOM issues)
        dim: dimensionality of latent space of VAE
    """

    def __init__(
        self,
        path_to_vae_statedict: str = "../vae/CEB_64.ckpt",
        dim: int = 64,
        init_data_timeout: int = 50,
        task_specific_args: list = [],  # list of additional args to be passed into objective funcion
        constraint_function_ids: list = [],  # list of strings identifying the black box constraint function to use
        constraint_thresholds: list = [],  # list of corresponding threshold values (floats)
        constraint_types: list = [],  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        init_csv_path: str = None,  # explicit path to a pre-built {x,y,censoring} init CSV; overrides the init_w_bao/init_w_random/init_w_llm fallback below when set. Additive: unset (None) reproduces every existing call site's behavior exactly.
        **kwargs,
    ):
        self.path_to_vae_statedict = path_to_vae_statedict
        self.dim = dim
        self.task_specific_args = task_specific_args
        self.init_data_timeout = init_data_timeout
        self.init_csv_path = init_csv_path
        # To specify constraints, pass in
        #   1. constraint_function_ids: a list of constraint function ids,
        #   2. constraint_thresholds: a list of thresholds,
        #   3. constraint_types: a list of threshold types (must be "min" or "max" for each)
        # Empty lists indicate that the problem is unconstrained, 1 or more constraints can be added
        assert len(constraint_function_ids) == len(constraint_thresholds)
        assert len(constraint_thresholds) == len(constraint_types)
        self.constraint_function_ids = (
            constraint_function_ids  # list of strings identifying the black box constraint function to use
        )
        self.constraint_thresholds = constraint_thresholds  # list of corresponding threshold values (floats)
        self.constraint_types = (
            constraint_types  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        )

        super().__init__(**kwargs)

        # add args to method args dict to be logged by wandb
        self.method_args["opt1"] = locals()
        del self.method_args["opt1"]["self"]

    def initialize_objective(self):
        # initialize objective
        self.objective = InfoTransformerVAEObjective(
            task_id=self.task_id,  # string id for your task
            task_specific_args=self.task_specific_args,  # list of additional args to be passed into objective funcion
            path_to_vae_statedict=self.path_to_vae_statedict,  # state dict for VAE to load in
            dim=self.dim,  # dimension of latent search space
            constraint_function_ids=self.constraint_function_ids,  # list of strings identifying the black box constraint function to use
            constraint_thresholds=self.constraint_thresholds,  # list of corresponding threshold values (floats)
            constraint_types=self.constraint_types,  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
            which_query_language=self.which_query_language,
            worst_runtime_observed=self.worst_runtime_observed,
            workload_name=self.workload_name,
            allow_cross_joins=self.allow_cross_joins,
            worst_init_x=self.init_train_x[self.init_train_y.argmin()],
        )
        # if train zs have not been pre-computed for particular vae, compute them
        #   by passing initialization selfies through vae
        try:
            self.init_train_z = self.compute_train_zs()
        except:
            # seed 0 randn was used to create init data so this works too
            set_seed(seed=0)
            N = len(self.init_train_x)
            self.init_train_z = torch.randn(N, self.dim)
            set_seed(seed=self.seed)
        # compute initial constriant values
        self.init_train_c = self.objective.compute_constraints(self.init_train_x)

        return self

    def compute_train_zs(self, bsz=32):
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
            self.init_censoring (a binary tensor indicating censoring for censored obs only)
        """
        if self.init_w_llm:  # init w/ BOLT's LLM generated queries
            # "150tasks_samples_temp07.jsonl"
            # "250tasks_samples_temp07.jsonl"
            init_data_path = f"../initialization_data/{self.init_w_llm_n_tasks}tasks_samples_temp07.jsonl"
            df = pd.read_json(init_data_path, lines=True, orient="records")
            df = df[df["task"] == self.workload_name]
            if df.shape[0] == 0:
                assert 0, f"no llm init data for workload {self.workload_name}"
            y = torch.tensor(df["time"].values[0]).float()  # torch.Size([50])
            # instead of setting n init, use all that llm generated
            self.num_initialization_points = y.shape[0]
            x = df["generated_answers"].values[0]
            assert y.max() < 0  # make sure run times in file are negated, otherwise censoring by <= -TIMEOUT is wrong
            # 1 --> censored, 0 --> uncensored
            censoring = np.zeros(y.shape)  # (50,)
            censoring[y <= -LLM_GEN_TIMEOUT] = censoring[y <= -LLM_GEN_TIMEOUT] + 1.0  # (50,)
            y = y.unsqueeze(-1)  # torch.Size([50, 1])
            censoring = torch.from_numpy(censoring).float()  # torch.Size([50])
            censoring = censoring.unsqueeze(-1)  # torch.Size([50, 1])
            self.init_censoring = censoring
            self.init_train_x = x
            self.init_train_y = y
            self.worst_runtime_observed = self.init_train_y.min().item() * -1
        elif self.init_w_bao:
            init_data_path = f"../initialization_data/bao_censored_initializations.csv"
            df = pd.read_csv(init_data_path, sep=";")  # (5537, 6)
            df = df[df["query_name"] == self.workload_name]  # (49, 6)
            xs_key = "plan"
            censoring_key = "censored"
            ceb_3k = False
            if df.shape[0] == 0:
                init_data_path = f"../initialization_data/ceb_3k_bao_initialization.csv"
                df = pd.read_csv(init_data_path)  # (146363, 8)
                df = df[df["query_name"] == self.workload_name]  # (49, 8)
                xs_key = "encoded_plan"
                censoring_key = "timed_out"
                ceb_3k = True
            if df.shape[0] == 0:
                assert 0, f"no bao init data for workload {self.workload_name}"

            y = df["runtime_secs"].values  # (49,)
            # instead of setting n init, use all that bao found
            self.num_initialization_points = y.shape[0]
            x = df[xs_key].values.tolist()
            x = [xi.split(",") for xi in x]
            x = [[int(xj) for xj in xi] for xi in x]  # len(x) 49
            censoring = df[censoring_key].values  # (49,)
            if not ceb_3k:
                censoring_temp = np.zeros(y.shape)  # (49,)
                # 1 --> censored, 0 --> uncensored
                censoring_temp[censoring] = censoring_temp[censoring] + 1.0
                censoring = censoring_temp
            y = torch.from_numpy(y).float()  # torch.Size([49])
            y = y.unsqueeze(-1)  # torch.Size([49, 1])
            if y.max() > 0:
                y = (
                    y * -1
                )  # if init data file did not already negate runtimes, negate here (create maximization problem)
            censoring = torch.from_numpy(censoring).float()  #  torch.Size([49])
            censoring = censoring.unsqueeze(-1)  # torch.Size([49, 1])
            self.init_censoring = censoring
            self.init_train_x = x
            self.init_train_y = y
            self.worst_runtime_observed = self.init_train_y.min().item() * -1
        elif self.init_w_random:
            init_data_path = f"../initialization_data/random_runs.csv"
            df = pd.read_csv(init_data_path, sep=";")  # (34557, 4)
            df = df[df["query_name"] == self.workload_name]  # (456, 4)
            y = df["runtime_secs"].values  # (456,)
            # instead of setting n init, use all that random found in the time limit
            self.num_initialization_points = y.shape[0]
            if self.num_initialization_points == 0:
                # No data available for JOB_33A/B/C
                print(f"Can't initialize with random init data for {self.workload_name}, no data available")
            x = df["plan"].values.tolist()
            x = [xi.split(",") for xi in x]
            x = [[int(xj) for xj in xi] for xi in x]  # len(x) 456
            censoring = np.zeros(y.shape)  # (456,)
            censoring_strings = df["result"].values  # (456,)
            # 1 --> censored, 0 --> uncensored
            censoring[censoring_strings == "timedout"] = censoring[censoring_strings == "timedout"] + 1.0
            y = torch.from_numpy(y).float()  # torch.Size([456])
            y = y.unsqueeze(-1)  # torch.Size([456, 1])
            if y.max() > 0:
                y = (
                    y * -1
                )  # if init data file did not already negate runtimes, negate here (create maximization problem)
            censoring = torch.from_numpy(censoring).float()  # torch.Size([456])
            censoring = censoring.unsqueeze(-1)  # torch.Size([456, 1])
            self.init_censoring = censoring
            self.init_train_x = x
            self.init_train_y = y
            self.worst_runtime_observed = self.init_train_y.min().item() * -1
        elif self.init_csv_path is not None:
            # Additive override (query_plan_experiment/steps.py::run_bo): a
            # pre-built, run-context-qualified init CSV, already scored by
            # the real oracle. Needed because none of init_w_bao/
            # init_w_random/init_w_llm above support a per-(experiment, arm,
            # workload) file -- e.g. held-out eval evaluates multiple
            # BOLT-<m> arms against the same 99 workloads, and the plain
            # `{workload_name}_init_data.csv` path below is shared by every
            # arm with no run-context qualifier, which would silently reuse
            # one arm's stale init data for the next. Falls through to the
            # exact same x/y/censoring extraction the else-branch below uses
            # (same {x, y, censoring} CSV schema) -- only init_data_path's
            # source differs.
            init_data_path = self.init_csv_path
            if not os.path.exists(init_data_path):
                raise FileNotFoundError(f"--init_csv_path={init_data_path} does not exist")

            df = pd.read_csv(init_data_path)

            x = df["x"].values.tolist()
            x = x[0 : self.num_initialization_points]

            # Preprocessing for databases stuff:
            x = [xi.split(",") for xi in x]
            x = [[int(xj) for xj in xi] for xi in x]

            y = torch.from_numpy(df["y"].values).float()
            y = y[0 : self.num_initialization_points]
            y = y.unsqueeze(-1)

            if y.max() > 0:
                y = (
                    y * -1
                )  # if init data file did not already negate runtimes, negate here (create maximization problem)
            self.init_train_x = x
            self.init_train_y = y
            self.worst_runtime_observed = self.init_train_y.min().item() * -1

            if self.censored_observations:
                cen = torch.from_numpy(df["censoring"].values)
                cen = cen[0 : self.num_initialization_points]
                cen = cen.unsqueeze(-1)
                self.init_censoring = cen
            else:
                self.init_censoring = None
        else:
            init_data_path = f"../initialization_data/{self.workload_name}_init_data.csv"

            if os.path.exists(init_data_path):
                df = pd.read_csv(init_data_path)
                x = df["x"].values.tolist()
                not_enough_init_data = len(x) < self.num_initialization_points

            if (not os.path.exists(init_data_path)) or not_enough_init_data:
                print("Creating and saving file of random init data points")
                create_init_data_aliases(
                    workload_name=self.workload_name,
                    timeout=self.init_data_timeout,
                    which_language=self.which_query_language,
                    N=self.num_initialization_points,
                    vae_latent_space_dim=self.dim,
                    path_to_vae=self.path_to_vae_statedict,
                )
                print(f"Initialization data saved, will load in this data for all future runs of {self.workload_name}")

            df = pd.read_csv(init_data_path)

            x = df["x"].values.tolist()
            x = x[0 : self.num_initialization_points]

            # Preprocessing for databases stuff:
            x = [xi.split(",") for xi in x]
            x = [[int(xj) for xj in xi] for xi in x]

            y = torch.from_numpy(df["y"].values).float()
            y = y[0 : self.num_initialization_points]
            y = y.unsqueeze(-1)

            if y.max() > 0:
                y = (
                    y * -1
                )  # if init data file did not already negate runtimes, negate here (create maximization problem)
            self.init_train_x = x
            self.init_train_y = y
            self.worst_runtime_observed = self.init_train_y.min().item() * -1

            if self.censored_observations:
                cen = torch.from_numpy(df["censoring"].values)
                cen = cen[0 : self.num_initialization_points]
                cen = cen.unsqueeze(-1)
                self.init_censoring = cen
            else:
                self.init_censoring = None

        return self


if __name__ == "__main__":
    fire.Fire(InfoTransformerVAEOptimization)

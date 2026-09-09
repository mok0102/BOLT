import torch

torch.set_float32_matmul_precision("highest")
import sys

sys.path.append("../")
import random
import numpy as np
import pandas as pd
import fire
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["WANDB_SILENT"] = "True"
from lolbo.lolbo import LOLBOState
from lolbo.latent_space_objective import LatentSpaceObjective
import signal
import copy
import time
from utils.set_seed import set_seed
from query_plan_timing import event, span

try:
    import wandb

    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False


class Optimize(object):
    """
    Run LOLBO Optimization
    Args:
        workload_name: name of workload for database query opt
        task_id: String id for optimization task, by default the wandb project name will be f'optimize-{task_id}'
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"optimimze-{self.task_id}")
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        max_non_parallel_runtime_hours: Max non-parallel runtime allowed allowed (budget). Optimization run terminates when this budget is exceeded (if None, use oracle calls instead)
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        update_e2e: If True, we update the models end to end (we run LOLBO). If False, we never update end to end (we run TuRBO)
        k: We keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc.
        censored_observations: Set to true if using an objective that sometimes givesn censored observations
        censored_obs_is_max: If True, the censored observaation is the max possible value (it could be lower). If False, the censored observaation is the MIN possible value (it could be higher)
        timeout_strategy: str, gives strategy to use to for timeout, must be one of (constant, percentile, gp)
        constant_timeout: int giving timeout in seconds if using constant timeout strategy
        timeout_percentile: float giving the percentile of gathered data to use to compute timeout for percentile timeout strategy
        gp_stdev_multiplier: num, value we muptiply gp stdev by to set new timeout with gp timeout strategy
        vanilla_bo: don't use trust regions for optimization (run vanilla bo)
    """

    def __init__(
        self,
        workload_name: str,
        task_id: str = "db",
        seed: int = None,
        track_with_wandb: bool = True,
        wandb_entity: str = "xxx",
        wandb_project_name: str = "",
        max_n_oracle_calls: int = 4_000,
        max_bo_steps: int = None,
        max_non_parallel_runtime_hours: int = None,
        learning_rte: float = 0.01,
        acq_func: str = "ts",
        bsz: int = 1,
        num_initialization_points: int = 200,
        init_n_update_epochs: int = 80,
        num_update_epochs: int = 2,
        e2e_freq: int = 10,
        update_e2e: bool = False,
        k: int = 100,
        verbose: bool = True,
        recenter_only=False,
        save_vae_ckpt=False,
        censored_observations=True,
        censored_obs_is_max=True,
        timeout_percentile=0.1,
        constant_timeout=1_000_000_000,  # for ~ no timeout baseline
        timeout_strategy="ours",
        gp_stdev_multiplier=2,
        which_query_language="aliases",
        vanilla_bo=False,
        save_freq=10,
        flag_correct_adding_vae_decode_time=True,
        allow_cross_joins=True,
        init_w_random=False,  # use random no-cross join data to init BO
        init_w_bao=True,  # use data from Bao to init BO
        init_w_llm=False,
        init_w_llm_n_tasks=150,
        eulbo=False,
        use_kg_eulbo=False,
        thompson_rejection=False,
    ):
        signal.signal(signal.SIGINT, self.handler)
        # add all local args to method args dict to be logged by wandb

        # can pick one of these three options or none, but not multiple (for ease of wandb filtering)
        assert not (init_w_bao and init_w_random)
        assert not (init_w_bao and init_w_llm)
        assert not (init_w_random and init_w_llm)
        self.init_w_random = init_w_random
        self.init_w_bao = init_w_bao
        self.init_w_llm = init_w_llm
        self.init_w_llm_n_tasks = init_w_llm_n_tasks

        if eulbo:
            torch.set_default_dtype(torch.float64)
            assert not update_e2e, "eulbo not implemented for e2e updates (lolbo) yet"

        self.eulbo = eulbo
        self.use_kg_eulbo = use_kg_eulbo
        self.allow_cross_joins = allow_cross_joins
        self.save_vae_ckpt = save_vae_ckpt
        self.flag_correct_adding_vae_decode_time = flag_correct_adding_vae_decode_time
        self.recenter_only = recenter_only  # only recenter, no E2E, unlike LOLBO
        self.method_args = {}
        self.method_args["init"] = locals()
        del self.method_args["init"]["self"]
        self.vanilla_bo = vanilla_bo
        self.which_query_language = which_query_language
        self.constant_timeout = constant_timeout
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity
        self.task_id = task_id
        self.workload_name = workload_name
        self.max_n_oracle_calls = max_n_oracle_calls
        if max_bo_steps is not None and max_bo_steps < 0:
            raise ValueError("max_bo_steps must be nonnegative")
        self.max_bo_steps = max_bo_steps
        self.max_non_parallel_runtime_hours = max_non_parallel_runtime_hours
        self.verbose = verbose
        self.num_initialization_points = num_initialization_points
        self.e2e_freq = e2e_freq
        self.censored_observations = censored_observations
        self.censored_obs_is_max = censored_obs_is_max
        self.update_e2e = update_e2e
        self.save_freq = save_freq
        self.k = k
        self.num_update_epochs = num_update_epochs
        self.init_n_update_epochs = init_n_update_epochs
        self.learning_rte = learning_rte
        self.bsz = bsz
        self.acq_func = acq_func
        self.timeout_percentile = timeout_percentile
        self.timeout_strategy = timeout_strategy
        self.gp_stdev_multiplier = gp_stdev_multiplier
        set_seed(seed)
        self.seed = seed
        self.thompson_rejection = thompson_rejection
        if wandb_project_name:  # if project name specified
            self.wandb_project_name = wandb_project_name
        else:  # otherwise use defualt
            self.wandb_project_name = f"optimimze-{self.task_id}"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, (
                "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"
            )

        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()

        # initialize train data for particular task
        #   must define self.init_train_x, self.init_train_y, and self.init_train_z
        with span("bo_initial_data_load_or_generate"):
            self.load_train_data()
        # initialize latent space objective (self.objective) for particular task
        with span("bo_objective_vae_initialization"):
            self.initialize_objective()

        assert isinstance(self.objective, LatentSpaceObjective), (
            "self.objective must be an instance of LatentSpaceObjective"
        )
        assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        if self.init_train_c is not None:  # if constrained
            assert torch.is_tensor(self.init_train_c), "load_train_data() must set self.init_train_c to a tensor of cs"
            assert self.init_train_c.shape[0] == len(self.init_train_x), (
                f"load_train_data() must initialize exactly the same number of cs and xs, instead got {len(self.init_train_x)} xs and {self.init_train_c.shape[0]} cs"
            )
        if self.censored_observations:
            assert torch.is_tensor(self.init_censoring), (
                "load_train_data() must set self.init_censoring to a binary tensor indicating which points are censored"
            )
            assert self.init_censoring.shape[0] == len(self.init_train_x), (
                f"load_train_data() must initialize exactly the same number of censoring binary indicationrs and xs, instead got {len(self.init_train_x)} xs and censroing with shape {self.init_censoring.shape}"
            )
        else:
            assert self.init_censoring is None
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"
        assert self.init_train_y.shape[0] == len(self.init_train_x), (
            f"load_train_data() must initialize exactly the same number of ys and xs, instead got {self.init_train_y.shape[0]} ys and {len(self.init_train_x)} xs"
        )
        assert self.init_train_z.shape[0] == len(self.init_train_x), (
            f"load_train_data() must initialize exactly the same number of zs and xs, instead got {self.init_train_z.shape[0]} zs and {len(self.init_train_x)} xs"
        )

        self.init_lolbo_state()

    def init_lolbo_state(self):
        # initialize lolbo state
        self.lolbo_state = LOLBOState(
            objective=self.objective,
            train_x=self.init_train_x,
            train_y=self.init_train_y,
            train_z=self.init_train_z,
            censoring=self.init_censoring,
            censored_obs_is_max=self.censored_obs_is_max,
            train_c=self.init_train_c,
            k=self.k,
            num_update_epochs=self.num_update_epochs,
            init_n_epochs=self.init_n_update_epochs,
            learning_rte=self.learning_rte,
            bsz=self.bsz,
            acq_func=self.acq_func,
            verbose=self.verbose,
            timeout_percentile=self.timeout_percentile,
            timeout_strategy=self.timeout_strategy,
            gp_stdev_multiplier=self.gp_stdev_multiplier,
            constant_timeout=self.constant_timeout,
            vanilla_bo=self.vanilla_bo,
            eulbo=self.eulbo,
            use_kg_eulbo=self.use_kg_eulbo,
            thompson_rejection=self.thompson_rejection,
        )
        return self

    def initialize_objective(self):
        """Initialize Objective for specific task
        must define self.objective object
        """
        return self

    def load_train_data(self):
        """Load in or randomly initialize self.num_initialization_points
        total initial data points to kick-off optimization
        Must define the following:
            self.init_train_x (a list of x's)
            self.init_train_y (a tensor of scores/y's)
            self.init_train_c (a tensor of constraint values/c's)
            self.init_censoring (a binary tensor indication if each point is censored, or None if not using censoring)
            self.init_train_z (a tensor of corresponding latent space points)
        """
        return self

    def create_wandb_tracker(self):
        if self.track_with_wandb:
            config_dict = {k: v for method_dict in self.method_args.values() for k, v in method_dict.items()}
            self.tracker = wandb.init(
                # project=self.wandb_project_name,
                # entity=self.wandb_entity,
                project='bolt',
                entity='jungmok',
                config=config_dict,
            )
            self.wandb_run_name = wandb.run.name
        else:
            self.tracker = None
            self.wandb_run_name = "no-wandb-tracking"

        return self

    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb:
            most_recent_batch_scores = self.lolbo_state.train_y[-self.lolbo_state.bsz :].squeeze()
            best_score_seen = self.lolbo_state.best_score_seen
            self.total_non_parallel_runtime_so_far += self.time_to_update_model
            self.total_non_parallel_runtime_so_far += self.lolbo_state.time_generate_candidates
            self.total_non_parallel_runtime_so_far += self.lolbo_state.time_set_oracle_timeout
            self.total_non_parallel_runtime_so_far += self.lolbo_state.time_update_dataset_w_new_points

            # OLD VERSION: combined vae decode time, oracle call time, and results organization time
            # self.total_non_parallel_runtime_so_far += self.lolbo_state.time_call_oracle_non_parallel

            # NEW VERSION, separate out those three and log separately:
            self.total_non_parallel_runtime_so_far += self.lolbo_state.time_vae_decode
            self.total_non_parallel_runtime_so_far += self.lolbo_state.time_query_oracle
            self.total_non_parallel_runtime_so_far += self.lolbo_state.time_organize_results
            self.running_total_oracle_query_time += self.lolbo_state.time_query_oracle

            total_non_parallel_runtime_so_far_hours = self.total_non_parallel_runtime_so_far / 3600
            dict_log = {
                "best_found": best_score_seen,
                "non_parallel_runtime": total_non_parallel_runtime_so_far_hours,
                "n_oracle_calls": self.lolbo_state.objective.num_calls,
                "total_number_of_e2e_updates": self.lolbo_state.tot_num_e2e_updates,
                "best_input_seen": self.lolbo_state.best_x_seen,
                "max_score_most_recent_batch": most_recent_batch_scores.max().item(),
                "min_score_most_recent_batch": most_recent_batch_scores.min().item(),
                "avg_score_most_recent_batch": most_recent_batch_scores.mean().item(),
                "time_to_update_model": self.time_to_update_model,
                "time_generate_candidates": self.lolbo_state.time_generate_candidates,
                "time_set_oracle_timeout": self.lolbo_state.time_set_oracle_timeout,
                "time_update_dataset_w_new_points": self.lolbo_state.time_update_dataset_w_new_points,
                "time_full_opt_loop": self.time_full_opt_loop,
                "num_surr_model_re_inits": self.lolbo_state.num_re_inits,
                "TR_length": self.lolbo_state.tr_state.length,
                # Old v: combined timings:
                # "time_call_oracle":self.lolbo_state.time_call_oracle,
                # "time_call_oracle_non_parallel":self.lolbo_state.time_call_oracle_non_parallel,
                # New v: separated out timings:
                "time_vae_decode": self.lolbo_state.time_vae_decode,
                "time_query_oracle": self.lolbo_state.time_query_oracle,  # NOTE: should NOT include wait in queue time!
                "time_organize_results": self.lolbo_state.time_organize_results,
                # also log queue wait time so we can check that out:
                "time_wait_in_oracle_queue": self.lolbo_state.time_wait_in_oracle_queue,
                "running_total_oracle_query_time": self.running_total_oracle_query_time,
            }
            self.tracker.log(dict_log)
            for tau in self.lolbo_state.timeouts_for_log:
                self.tracker.log(
                    {
                        "timeout": tau,
                        "best_per_tau": best_score_seen,
                        "non_parallel_runtime_per_tau": total_non_parallel_runtime_so_far_hours,
                    }
                )

        return self

    def run_lolbo(self):
        """Main optimization loop"""
        # record timings of following, initialize to 0
        self.total_non_parallel_runtime_so_far = 0
        self.running_total_oracle_query_time = 0
        self.time_to_update_model = 0
        self.lolbo_state.time_generate_candidates = 0
        self.lolbo_state.time_set_oracle_timeout = 0

        # Old v: combined timings vae deocde, call oracle, and orgaize results
        # self.lolbo_state.time_call_oracle = 0
        # self.lolbo_state.time_call_oracle_non_parallel = 0
        # New v: seperated timings
        self.lolbo_state.time_organize_results = 0
        self.lolbo_state.time_vae_decode = 0
        self.lolbo_state.time_query_oracle = 0
        self.lolbo_state.time_wait_in_oracle_queue = 0

        self.lolbo_state.time_update_dataset_w_new_points = 0
        self.time_full_opt_loop = 0
        n_iters = 0
        # main optimization loop
        contine_run_condition = True
        while contine_run_condition:
            if self.max_bo_steps is not None and n_iters >= self.max_bo_steps:
                break
            start_full_opt_loop_time = time.time()
            self.log_data_to_wandb_on_each_loop()
            if self.max_non_parallel_runtime_hours is None:
                contine_run_condition = self.lolbo_state.objective.num_calls < self.max_n_oracle_calls
            else:
                total_non_parallel_runtime_so_far_hours = self.total_non_parallel_runtime_so_far / 3600
                contine_run_condition = total_non_parallel_runtime_so_far_hours < self.max_non_parallel_runtime_hours
            if not contine_run_condition:
                break
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            start_update_models = time.time()
            if (self.lolbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                if not self.recenter_only:
                    self.lolbo_state.update_models_e2e()
                self.lolbo_state.recenter()
                if self.recenter_only:
                    self.lolbo_state.update_surrogate_model()
            else:  # otherwise, just update the surrogate model on data
                self.lolbo_state.update_surrogate_model()
            self.time_to_update_model = time.time() - start_update_models
            event("bo_surrogate_update", self.time_to_update_model, iteration=n_iters)
            # generate new candidate points, evaluate them, and update data
            with span("bo_acquisition_including_db", iteration=n_iters):
                self.lolbo_state.acquisition()
            for phase, attr in [("bo_candidate_generation", "time_generate_candidates"),
                                ("bo_timeout_selection", "time_set_oracle_timeout"),
                                ("bo_vae_decode", "time_vae_decode")]:
                event(phase, getattr(self.lolbo_state, attr, 0.0), iteration=n_iters)
            # if a new best has been found, print out new best input and score:
            if self.lolbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.lolbo_state.new_best_found = False
            if (n_iters % self.save_freq) == 0:  # save all collected data every save_freq iterations
                self.save_all_collected_data()
            self.time_full_opt_loop = time.time() - start_full_opt_loop_time
            n_iters += 1

        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        # save all data collected during optimization
        self.save_all_collected_data()
        if self.tracker is not None:
            self.tracker.finish()

        return self

    def print_progress_update(self):
        """Important data printed each time a new
        best input is found, as well as at the end
        of the optimization run
        (only used if self.verbose==True)
        More print statements can be added her as desired
        """
        if self.track_with_wandb:
            print(f"Optimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.lolbo_state.best_x_seen}")
        print(f"Best {self.workload_name} Score: {self.lolbo_state.best_score_seen}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.lolbo_state.objective.num_calls}")

        return self

    def handler(self, signum, frame):
        # if we Ctrl-c, make sure we log top xs, scores found
        # try:
        #     deprovision_instances_all_tasks(wandb.run.name)
        #     # deprovision_instances(wandb.run.name)
        #     print("Successfully Deprovisioned EC2 Instances")
        # except:
        #     print("Failed to Deprovision EC2 Instances, pleae do manually if relevant")
        print("Ctrl-c hass been pressed, wait while we save all collected data...")
        self.save_all_collected_data()
        print("Now terminating wandb tracker...")
        if self.tracker is not None:
            self.tracker.finish()
        msg = "Data now saved and tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)

    def save_all_collected_data(self):
        """After optimization finishes, save all collected data locally"""
        save_dir = "optimization_all_collected_data/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if self.track_with_wandb:
            wandb_run_name1 = wandb.run.name
        else:
            wandb_run_name1 = "nowandb"
        file_path = (
            save_dir + self.wandb_project_name + "_" + wandb_run_name1 + f"_{self.workload_name}_all-data-collected.csv"
        )
        df = {}
        train_x_save = self.lolbo_state.train_x
        if type(train_x_save[0]) == list:
            train_x_save = [str(x) for x in train_x_save]
        df["train_x"] = np.array(train_x_save)
        df["train_y"] = self.lolbo_state.train_y.squeeze().detach().cpu().numpy()
        if self.censored_observations:
            df["censoring"] = self.lolbo_state.censoring.squeeze().detach().cpu().numpy()
        df = pd.DataFrame.from_dict(df)
        df.to_csv(file_path, index=None)

        return self

    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)


if __name__ == "__main__":
    fire.Fire(Optimize)

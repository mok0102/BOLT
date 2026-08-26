import numpy as np
import torch
import sys

sys.path.append("../")
from lolbo.latent_space_objective import LatentSpaceObjective
from vae.model import VAEModule
from your_tasks.your_objective_functions import OBJECTIVE_FUNCTIONS_DICT
from your_tasks.your_blackbox_constraints import CONSTRAINT_FUNCTIONS_DICT
from oracle.oracle import plan_has_crossjoin


class InfoTransformerVAEObjective(LatentSpaceObjective):
    """Objective class for latent space objectives using InfoTransformerVAE"""

    def __init__(
        self,
        task_id="db",
        task_specific_args=[],
        path_to_vae_statedict="../vae/64.ckpt",
        dim=16,  # dimension of latent search space
        init_vae=True,  # whether or not to even initialize VAE (avoid this step to save time if we just wnat to eval oracle on xs directly)
        constraint_function_ids=[],  # list of strings identifying the black box constraint function to use
        constraint_thresholds=[],  # list of corresponding threshold values (floats)
        constraint_types=[],  # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        xs_to_scores_dict={},
        xs_to_censoring_dict={},
        num_calls=0,
        shared_vae=None,
        which_query_language="aliases",
        worst_runtime_observed=200,
        workload_name="CEB_1A10",
        allow_cross_joins=True,
        worst_init_x=None,
    ):
        self.dim = dim
        self.path_to_vae_statedict = path_to_vae_statedict
        self.task_specific_args = task_specific_args
        self.shared_vae = shared_vae
        self.allow_cross_joins = allow_cross_joins
        self.worst_init_x = worst_init_x
        self.objective_function = OBJECTIVE_FUNCTIONS_DICT[task_id](
            which_language=which_query_language,
            worst_runtime_observed=worst_runtime_observed,
            workload_name=workload_name,
        )
        if not allow_cross_joins:
            assert worst_init_x is not None

        self.constraint_functions = []
        for ix, constraint_threshold in enumerate(constraint_thresholds):
            cfunc_class = CONSTRAINT_FUNCTIONS_DICT[constraint_function_ids[ix]]
            cfunc = cfunc_class(
                threshold_value=constraint_threshold,
                threshold_type=constraint_types[ix],
            )
            self.constraint_functions.append(cfunc)

        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            xs_to_censoring_dict=xs_to_censoring_dict,
            task_id=task_id,
            init_vae=init_vae,
        )

    def vae_decode(self, z):
        """Input
            z: a tensor latent space points (bsz, self.dim)
        Output
            a corresponding list of the decoded input space
            items output by vae decoder
        """
        if type(z) is np.ndarray:
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()
        samples = self.vae.sample(z=z)  # =z.reshape(-1, 2, self.dim//2)
        if not self.allow_cross_joins:
            good_samples = []
            for ix, sample in enumerate(samples):
                has_cross_joins = self.has_cross_joins(query_plan=sample)
                n_fails = 0
                while has_cross_joins:
                    sample = self.vae.sample(z=z[ix, :])[0]
                    has_cross_joins = self.has_cross_joins(query_plan=sample)
                    n_fails += 1
                    if n_fails > 50:
                        sample = self.worst_init_x  # just something random we know doesn't have cross joins
                        break
                good_samples.append(sample)

            samples = good_samples

        return samples

    def has_cross_joins(self, query_plan):
        return plan_has_crossjoin(workload=self.objective_function.full_workload_spec, encoded=query_plan)

    def query_oracle(self, x, timeouts_list=None):
        if timeouts_list is None:
            scores_list, censoring_list = self.objective_function(x)
        else:
            scores_list, censoring_list = self.objective_function(x, timeouts_list)
        return scores_list, censoring_list

    def initialize_vae(self):
        """Sets self.vae to the desired pretrained vae"""
        if self.shared_vae is not None:
            # for case when multiple runs share the same vae, don't want to
            # re-init the same vae multiple times
            self.vae = self.shared_vae
        elif not self.path_to_vae_statedict:
            # Cold-start (query_plan_experiment): the paper's real VAE
            # (appendix C.1) is a pre-trained, never-retrained-during-BO
            # artifact from external prior work (Tao et al. 2025) that isn't
            # available anywhere in this repo -- only a placeholder
            # checkpoint file exists. Construct a randomly-initialized
            # VAEModule instead (bn_size=1 so its latent dim is exactly
            # self.dim, matching every caller that assumes a self.dim-sized
            # z e.g. random z fallback / vae_forward's reshape). A documented,
            # known fidelity deviation -- see
            # imp_plan/02_query_plan_reimplementation_plan.md. codec.py's
            # decode already tolerates a nominal (table, alias) pair outside
            # a specific query's real alias range via hash-mod resolution,
            # so num_aliases only needs to be a generous, self-consistent
            # vocabulary bound, not exactly right for every query.
            from vae.model import build_vocab

            all_tables = self.objective_function.full_workload_spec.all_tables
            max_query_aliases = max(
                (n for _, n in self.objective_function.full_workload_spec.query_tables), default=1
            )
            vocab, rev_vocab = build_vocab(num_tables=len(all_tables), num_aliases=max_query_aliases + 4)
            self.vae = VAEModule(vocab=vocab, rev_vocab=rev_vocab, bn_size=1, d_neck=self.dim).cuda()
        else:
            self.vae = VAEModule.load_from_checkpoint(self.path_to_vae_statedict).cuda()

    def vae_forward(self, xs_batch):
        """Input:
            a list xs
        Output:
            z: tensor of resultant latent space codes
                obtained by passing the xs through the encoder
            vae_loss: the total loss of a full forward pass
                of the batch of xs through the vae
                (ie reconstruction error)
        """
        # assumes xs_batch is a batch of smiles strings
        # tokenized_seqs = self.dataobj.tokenize_sequence(xs_batch)
        # encoded_seqs = [self.dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
        # X = collate_fn(encoded_seqs)
        dict1 = self.vae(xs_batch)
        vae_loss, z = dict1["loss"], dict1["z"]
        z = z.reshape(-1, self.dim)  # Necessary?

        return z, vae_loss

    # black box constraint, treat as oracle
    @torch.no_grad()
    def compute_constraints(self, xs_batch):
        """Input:
            a list xs (list of sequences)
        Output:
            c: tensor of size (len(xs),n_constraints) of
                resultant constraint values, or
                None of problem is unconstrained
                Note: constraints, must be of form c(x) <= 0!
        """
        if len(self.constraint_functions) == 0:
            return None

        all_cvals = []
        for cfunc in self.constraint_functions:
            cvals = cfunc(xs_batch)
            all_cvals.append(cvals)

        return torch.cat(all_cvals, -1)

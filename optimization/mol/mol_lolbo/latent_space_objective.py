import numpy as np
import torch


class LatentSpaceObjective:
    """Base class for any latent space optimization task
    class supports any optimization task with accompanying VAE
    such that during optimization, latent space points (z)
    must be passed through the VAE decoder to obtain
    original input space points (x) which can then
    be passed into the oracle to obtain objective values (y)"""

    def __init__(
        self,
        xs_to_scores_dict=None,
        num_calls=0,
        task_id="",
        init_vae=True,
    ):
        # dict used to track xs and scores (ys) queried during optimization
        if xs_to_scores_dict is None:
            xs_to_scores_dict = {}
        self.xs_to_scores_dict = xs_to_scores_dict

        # track total number of times the oracle has been called
        self.num_calls = num_calls

        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id

        # load in pretrained VAE, store in variable self.vae
        self.vae = None
        if init_vae:
            self.initialize_vae()
            assert self.vae is not None

    def __call__(self, z):
        """Input
            z: a numpy array or pytorch tensor of latent space points
        Output
            out_dict['valid_zs'] = the zs which decoded to valid xs
            out_dict['decoded_xs'] = an array of valid xs obtained from input zs
            out_dict['scores']: an array of valid scores obtained from input zs
            out_dict['constr_vals']: an array of constraint values or none if unconstrained
        """
        if type(z) is np.ndarray:
            z = torch.from_numpy(z).float()
        decoded_xs = self.vae_decode(z)
        scores = []
        xs_to_be_queired = []
        for x in decoded_xs:
            # get rid of X's (deletion)
            # if we have already computed the score, don't
            #   re-compute (don't call oracle unnecessarily)
            if x in self.xs_to_scores_dict:
                score = self.xs_to_scores_dict[x]
            else:  # otherwise call the oracle to get score
                score = "?"
                xs_to_be_queired.append(x)
            scores.append(score)

        computed_scores = self.query_oracle(xs_to_be_queired)
        # move computed scores to scores list
        temp = []
        ix = 0
        for score in scores:
            if score == "?":
                temp.append(computed_scores[ix])
                # add score to dict so we don't have to
                #   compute it again if we get the same input x
                self.xs_to_scores_dict[xs_to_be_queired[ix]] = computed_scores[ix]
                ix += 1
            else:
                temp.append(score)
        scores = temp

        # track number of oracle calls
        #   nan scores happen when we pass an invalid
        #   molecular string and thus avoid calling the
        #   oracle entirely
        self.num_calls += (np.logical_not(np.isnan(np.array(computed_scores)))).sum()

        scores_arr = np.array(scores)
        decoded_xs = np.array(decoded_xs)
        # get valid zs, xs, and scores
        bool_arr = np.logical_not(np.isnan(scores_arr))
        decoded_xs = decoded_xs[bool_arr]
        scores_arr = scores_arr[bool_arr]
        valid_zs = z[bool_arr]

        out_dict = {}
        out_dict["scores"] = scores_arr
        out_dict["valid_zs"] = valid_zs
        out_dict["decoded_xs"] = decoded_xs
        out_dict["constr_vals"] = self.compute_constraints(decoded_xs)

        return out_dict

    def vae_decode(self, z):
        """Input
            z: a tensor latent space points
        Output
            a corresponding list of the decoded input space
            items output by vae decoder
        """
        raise NotImplementedError("Must implement vae_decode()")

    def query_oracle(self, x):
        """Input:
            a list of input space items x (i.e. molecule strings)
        Output:
            method queries the oracle and returns
            the corresponding list of scores y,
            or np.nan in the case that x is an invalid input
        """
        raise NotImplementedError(
            "Must implement query_oracle() specific to desired optimization task"
        )

    def initialize_vae(self):
        """Sets variable self.vae to the desired pretrained vae"""
        raise NotImplementedError(
            "Must implement method initialize_vae() to load in vae for desired optimization task"
        )

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
        raise NotImplementedError(
            "Must implement method vae_forward() (forward pass of vae)"
        )

    def compute_constraints(self, xs_batch):
        """Input:
            a list xs
        Output:
            c: tensor of size (len(xs),n_constraints) of
                resultant constraint values, or
                None of problem is unconstrained
                Note: constraints, must be of form c(x) <= 0!
        """
        return None

import json
import math
from pathlib import Path

import gpytorch
import numpy as np
import torch
from gpytorch.mlls import PredictiveLogLikelihood

import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_dir)
sys.path.append(f"{parent_dir}")

from lolbo.utils.bo_utils.gp_map_saas import MAPSaasGPModel
from lolbo.utils.bo_utils.poe_gp import PoEGPModel
from lolbo.utils.bo_utils.ppgpr import GPModelDKL
from lolbo.utils.bo_utils.turbo import TurboState, generate_batch, update_state
from lolbo.utils.utils import (
    update_constraint_surr_models,
    update_models_end_to_end_with_constraints,
    update_surr_model,
)


class LOLBOState:
    def __init__(
        self,
        objective,
        surrogate_type: str,
        train_x,
        train_y,
        train_z,
        train_c=None,
        k=1_000,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        acq_func="ts",
        verbose=True,
        pretrained_surrogate_path: str | None = None,
        pretrained_surrogate_num_inducing: int | None = None,
        poe_manifest_path: str | None = None,
    ):
        self.objective = objective  # objective with vae for particular task
        self.train_x = train_x  # initial train x data
        self.train_y = train_y  # initial train y data
        self.train_z = train_z  # initial train z data
        self.train_c = train_c  # initial constraint values data
        self.minimize = minimize  # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k = k  # track and update on top k scoring points found
        self.num_update_epochs = num_update_epochs  # num epochs update models
        self.init_n_epochs = (
            init_n_epochs  # num epochs train surr model on initial data
        )
        self.learning_rte = learning_rte  # lr to use for model updates
        self.bsz = bsz  # acquisition batch size
        self.acq_func = acq_func  # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose = verbose
        self.surrogate_type = surrogate_type
        # None (default): unchanged behavior, a fresh GPModelDKL is fit from
        # scratch on this task's own train_z, same as ever. A path (shared-
        # surrogate MTBO baseline): initialize_surrogate_model() below loads
        # this state dict into the objective model instead -- the inducing-
        # point count is baked into its shapes, so
        # pretrained_surrogate_num_inducing must be given alongside it and
        # must match exactly what the checkpoint was trained with.
        self.pretrained_surrogate_path = pretrained_surrogate_path
        self.pretrained_surrogate_num_inducing = pretrained_surrogate_num_inducing
        # GP-expert-transfer baseline (POGPE/SGPE): a path to a JSON manifest
        # of {"experts": [{"path": ..., "weight": ...}, ...]} -- an ensemble
        # of K frozen pretrained experts, combined via product-of-experts
        # (lolbo/utils/bo_utils/poe_gp.py::PoEGPModel) instead of a single
        # surrogate. Mutually exclusive with pretrained_surrogate_path in
        # practice (surrogate_type="gp_poe" vs "gp_dkl"), kept as a separate
        # field since the shapes genuinely differ (one path+one inducing
        # count vs. a manifest of many).
        self.poe_manifest_path = poe_manifest_path

        assert acq_func in ["ei", "ts"]
        if minimize:
            self.train_y = self.train_y * -1

        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        # self.best_score_seen = torch.max(train_y)
        # self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = (
            False  # initial training of surrogate model uses all data for more epochs
        )
        self.new_best_found = False

        self.initialize_top_k()
        self.initialize_surrogate_model()
        self.initialize_tr_state()
        self.initialize_xs_to_scores_dict()

    def initialize_xs_to_scores_dict(
        self,
    ):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict

    def initialize_top_k(self):
        """Initialize top k x, y, and zs"""
        # if we have constriants, the top k are those that meet constraints!
        if self.train_c is not None:
            bool_arr = torch.all(
                self.train_c <= 0, dim=-1
            )  # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_train_z = self.train_z[bool_arr]
            valid_train_x = np.array(self.train_x)[bool_arr]
            valid_train_c = self.train_c[bool_arr]
        else:
            vaid_train_y = self.train_y
            valid_train_z = self.train_z
            valid_train_x = self.train_x

        if len(vaid_train_y) > 1:
            self.best_score_seen = torch.max(vaid_train_y)
            self.best_x_seen = valid_train_x[torch.argmax(vaid_train_y.squeeze())]

            # track top k scores found
            self.top_k_scores, top_k_idxs = torch.topk(
                vaid_train_y.squeeze(), min(self.k, vaid_train_y.shape[0])
            )
            self.top_k_scores = self.top_k_scores.tolist()
            top_k_idxs = top_k_idxs.tolist()
            self.top_k_xs = [valid_train_x[i] for i in top_k_idxs]
            self.top_k_zs = [valid_train_z[i].unsqueeze(-2) for i in top_k_idxs]
            if self.train_c is not None:
                self.top_k_cs = [valid_train_c[i].unsqueeze(-2) for i in top_k_idxs]
        elif len(vaid_train_y) == 1:
            self.best_score_seen = vaid_train_y.item()
            self.best_x_seen = valid_train_x.item()
            self.top_k_scores = [self.best_score_seen]
            self.top_k_xs = [self.best_x_seen]
            self.top_k_zs = [valid_train_z]
            if self.train_c is not None:
                self.top_k_cs = [valid_train_c]
        else:
            print("No valid init data according to constraint(s)")
            self.best_score_seen = None
            self.best_x_seen = None
            self.top_k_scores = []
            self.top_k_xs = []
            self.top_k_zs = []
            if self.train_c is not None:
                self.top_k_cs = []

    def initialize_tr_state(self):
        if self.train_c is not None:  # if constrained
            bool_arr = torch.all(
                self.train_c <= 0, dim=-1
            )  # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_c_vals = self.train_c[bool_arr]
        else:
            vaid_train_y = self.train_y
            best_constraint_values = None

        if len(vaid_train_y) == 0:
            best_value = -torch.inf
            if self.minimize:
                best_value = torch.inf
            if self.train_c is not None:
                best_constraint_values = (
                    torch.ones(1, self.train_c.shape[1]) * torch.inf
                )
        else:
            best_value = torch.max(vaid_train_y).item()
            if self.train_c is not None:
                best_constraint_values = valid_c_vals[torch.argmax(vaid_train_y)]
                if len(best_constraint_values.shape) == 1:
                    best_constraint_values = best_constraint_values.unsqueeze(-1)
        # initialize turbo trust region state
        self.tr_state = TurboState(  # initialize turbo state
            dim=self.train_z.shape[-1],
            batch_size=self.bsz,
            best_value=best_value,
            best_constraint_values=best_constraint_values,
        )

        return self

    def initialize_constraint_surrogates(self):
        self.c_models = []
        self.c_mlls = []
        for _i in range(self.train_c.shape[1]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            n_pts = min(self.train_z.shape[0], 1024)
            c_model = GPModelDKL(
                self.train_z[:n_pts, :].cuda(), likelihood=likelihood
            ).cuda()
            c_mll = PredictiveLogLikelihood(
                c_model.likelihood, c_model, num_data=self.train_z.size(-2)
            )
            c_model = c_model.eval()
            # c_model = self.model.cuda()
            self.c_models.append(c_model)
            self.c_mlls.append(c_mll)
        return self

    def initialize_surrogate_model(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        n_pts = min(self.train_z.shape[0], 1024)
        construction_z = self.train_z
        if self.pretrained_surrogate_path is not None:
            assert self.pretrained_surrogate_num_inducing is not None, (
                "pretrained_surrogate_num_inducing must be given alongside pretrained_surrogate_path "
                "-- the inducing-point count is baked into the saved state dict's shapes and must match exactly"
            )
            n_pts = self.pretrained_surrogate_num_inducing
            if self.train_z.shape[0] < n_pts:
                # GPModelDKL's constructor only uses these values to fix tensor SHAPES
                # (inducing-point count, feature-extractor dims) -- every actual value
                # (inducing points, feature-extractor weights, variational params) gets
                # completely overwritten by load_state_dict() below. So a held-out init
                # pool smaller than the pretrained inducing-point count is not actually
                # a hard requirement -- pad by repeat-sampling existing rows purely to
                # reach the right construction shape (real bug found via a real-scale
                # run: MTBO's checkpoints use 1024 inducing points, but held-out target
                # pool sizes as small as 10 are a legitimate, intended comparison point).
                pad_idx = torch.randint(0, self.train_z.shape[0], (n_pts - self.train_z.shape[0],))
                construction_z = torch.cat([self.train_z, self.train_z[pad_idx]], dim=0)
        if self.surrogate_type == "gp_dkl":
            print("Using GP DKL surrogate model")
            self.model = GPModelDKL(
                construction_z[:n_pts, :].cuda(), likelihood=likelihood
            ).cuda()
            if self.pretrained_surrogate_path is not None:
                print(f"Loading pretrained surrogate state dict from {self.pretrained_surrogate_path}")
                state_dict = torch.load(self.pretrained_surrogate_path)
                self.model.load_state_dict(state_dict, strict=True)
        elif self.surrogate_type == "gp_saas":
            print("Using MAP SaaS GP surrogate model")
            self.model = MAPSaasGPModel(
                self.train_z[:n_pts, :].cuda(), likelihood=likelihood
            ).cuda()
        elif self.surrogate_type == "gp_poe":
            print(f"Using PoE ensemble surrogate model, manifest={self.poe_manifest_path}")
            self.model = self._build_poe_model()
        else:
            raise ValueError(f"Surrogate type {self.surrogate_type} not recognized")

        if self.surrogate_type == "gp_poe":
            # PoEGPModel wraps K frozen experts, each with its own likelihood --
            # there is no single shared mll to compute for the ensemble as a
            # whole. update_surrogate_model() below is a no-op for this
            # surrogate_type, so self.mll is never actually used, but every
            # other code path (e.g. update_models_e2e's signature) expects the
            # attribute to exist.
            self.mll = None
        else:
            self.mll = PredictiveLogLikelihood(
                self.model.likelihood, self.model, num_data=self.train_z.size(-2)
            )
        self.model = self.model.eval()
        self.model = self.model.cuda()

        if self.train_c is not None:
            self.initialize_constraint_surrogates()

        return self

    def _build_poe_model(self) -> PoEGPModel:
        """Loads every expert listed in self.poe_manifest_path into a fresh
        GPModelDKL (same load pattern as the single-pretrained-surrogate
        branch above, including the same inducing-point-padding fix -- each
        expert's own num_inducing_points is baked into its own state dict's
        shapes, read from its sidecar surrogate_meta.json, and may itself
        exceed this held-out task's train_z size)."""
        manifest = json.loads(Path(self.poe_manifest_path).read_text())
        experts, weights = [], []
        for entry in manifest["experts"]:
            expert_path = Path(entry["path"])
            expert_meta = json.loads(expert_path.with_name("surrogate_meta.json").read_text())
            n_ind = expert_meta["num_inducing_points"]
            expert_construction_z = self.train_z
            if self.train_z.shape[0] < n_ind:
                pad_idx = torch.randint(0, self.train_z.shape[0], (n_ind - self.train_z.shape[0],))
                expert_construction_z = torch.cat([self.train_z, self.train_z[pad_idx]], dim=0)
            expert_likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            expert_model = GPModelDKL(
                expert_construction_z[:n_ind, :].cuda(), likelihood=expert_likelihood
            ).cuda()
            expert_model.load_state_dict(torch.load(expert_path), strict=True)
            experts.append(expert_model)
            weights.append(entry["weight"])
        return PoEGPModel(experts, weights)

    def update_next(self, z_next_, y_next_, x_next_, c_next_=None, acquisition=False):
        """Add new points (z_next, y_next, x_next) to train data
        and update progress (top k scores found so far)
        and update trust region state
        """

        if c_next_ is not None:
            if len(c_next_.shape) == 1:
                c_next_ = c_next_.unsqueeze(-1)
            valid_points = torch.all(c_next_ <= 0, dim=-1)  # all constraint values <= 0
        else:
            valid_points = torch.tensor([True] * len(y_next_))
        z_next_ = z_next_.detach().cpu()
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze()
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        progress = False
        for i, score in enumerate(y_next_):
            self.train_x.append(x_next_[i])
            if valid_points[i]:  # if y is valid according to constraints
                if len(self.top_k_scores) < self.k:
                    # if we don't yet have k top scores, add it to the list
                    self.top_k_scores.append(score.item())
                    self.top_k_xs.append(x_next_[i])
                    self.top_k_zs.append(z_next_[i].unsqueeze(-2))
                    if (
                        self.train_c is not None
                    ):  # if constrained, update best constraints too
                        self.top_k_cs.append(c_next_[i].unsqueeze(-2))
                elif score.item() > min(self.top_k_scores) and (
                    x_next_[i] not in self.top_k_xs
                ):
                    # if the score is better than the worst score in the top k list, upate the list
                    min_score = min(self.top_k_scores)
                    min_idx = self.top_k_scores.index(min_score)
                    self.top_k_scores[min_idx] = score.item()
                    self.top_k_xs[min_idx] = x_next_[i]
                    self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2)  # .cuda()
                    if (
                        self.train_c is not None
                    ):  # if constrained, update best constraints too
                        self.top_k_cs[min_idx] = c_next_[i].unsqueeze(-2)
                # if this is the first valid example we've found, OR if we imporve
                if (self.best_score_seen is None) or (
                    score.item() > self.best_score_seen
                ):
                    self.progress_fails_since_last_e2e = 0
                    progress = True
                    self.best_score_seen = score.item()  # update best
                    self.best_x_seen = x_next_[i]
                    self.new_best_found = True
        if (
            not progress
        ) and acquisition:  # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        if acquisition:
            self.tr_state = update_state(
                state=self.tr_state,
                Y_next=y_next_,
                C_next=c_next_,
            )
        self.train_z = torch.cat((self.train_z, z_next_), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next_), dim=-2)
        if c_next_ is not None:
            self.train_c = torch.cat((self.train_c, c_next_), dim=-2)

        return self

    def update_surrogate_model(self):
        if self.surrogate_type == "gp_poe":
            # The ensemble is frozen for the whole held-out run by design --
            # POGPE/SGPE's whole point is K pretrained experts (+ SGPE's one
            # online target expert, fit once up front) reused as-is, not
            # retrained every acquisition round. Retraining all K+1 members
            # every round would also reproduce exactly the "must query/
            # maintain K models at every step" cost blowup the source paper
            # flags -- here we only ever pay the query cost, never a
            # per-step retrain cost.
            return self
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            train_z = self.train_z
            train_y = self.train_y.squeeze(-1)
            train_c = self.train_c
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            train_z = self.train_z[-self.bsz :]
            train_y = self.train_y[-self.bsz :].squeeze(-1)
            if self.train_c is not None:
                train_c = self.train_c[-self.bsz :]
            else:
                train_c = None

        self.model = update_surr_model(
            self.model, self.mll, self.learning_rte, train_z, train_y, n_epochs
        )
        if self.train_c is not None:
            self.c_models = update_constraint_surr_models(
                self.c_models,
                self.c_mlls,
                self.learning_rte,
                train_z,
                train_c,
                n_epochs,
            )

        self.initial_model_training_complete = True

        return self

    def update_models_e2e(self):
        """Finetune VAE end to end with surrogate model"""
        if self.surrogate_type == "gp_poe":
            raise NotImplementedError(
                "end-to-end VAE fine-tuning is not supported for the gp_poe surrogate type "
                "(a frozen K-expert ensemble has no single model/mll to fine-tune against) -- "
                "callers must pass update_e2e=False whenever surrogate_type='gp_poe'"
            )
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.bsz :]
        new_ys = self.train_y[-self.bsz :].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()

        c_models = []
        c_mlls = []
        train_c = None
        if self.train_c is not None:
            c_models = self.c_models
            c_mlls = self.c_mlls
            new_cs = self.train_c[-self.bsz :]
            # Note: self.top_k_cs is a list of (1, n_cons) tensors
            if len(self.top_k_cs) > 0:
                top_k_cs_tensor = torch.cat(self.top_k_cs, -2).float()
                train_c = torch.cat((new_cs, top_k_cs_tensor), -2).float()
            else:
                train_c = new_cs
            # train_c = torch.tensor(new_cs + self.top_k_cs).float()

        self.objective, self.model = update_models_end_to_end_with_constraints(
            train_x=train_x,
            train_y_scores=train_y,
            objective=self.objective,
            model=self.model,
            mll=self.mll,
            learning_rte=self.learning_rte,
            num_update_epochs=self.num_update_epochs,
            train_c_scores=train_c,
            c_models=c_models,
            c_mlls=c_mlls,
        )
        self.tot_num_e2e_updates += 1

        return self

    def recenter(self):
        """Pass SELFIES strings back through
        VAE to find new locations in the
        new fine-tuned latent space
        """
        self.objective.vae.eval()
        self.model.train()

        optimize_list = [{"params": self.model.parameters(), "lr": self.learning_rte}]
        if self.train_c is not None:
            for c_model in self.c_models:
                c_model.train()
                optimize_list.append(
                    {"params": c_model.parameters(), "lr": self.learning_rte}
                )
        optimizer1 = torch.optim.Adam(optimize_list, lr=self.learning_rte)
        new_xs = self.train_x[-self.bsz :]
        train_x = new_xs + self.top_k_xs
        max_string_len = len(max(train_x, key=len))
        # max batch size smaller to avoid memory limit
        #   with longer strings (more tokens)
        bsz = max(1, int(2560 / max_string_len))
        num_batches = math.ceil(len(train_x) / bsz)
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                optimizer1.zero_grad()
                with torch.no_grad():
                    start_idx, stop_idx = batch_ix * bsz, (batch_ix + 1) * bsz
                    batch_list = train_x[start_idx:stop_idx]
                    z, _ = self.objective.vae_forward(batch_list)
                    out_dict = self.objective(z)
                    scores_arr = out_dict["scores"]
                    constraints_tensor = out_dict["constr_vals"]
                    valid_zs = out_dict["valid_zs"]
                    xs_list = out_dict["decoded_xs"]
                if len(scores_arr) > 0:  # if some valid scores
                    scores_arr = torch.from_numpy(scores_arr)
                    if self.minimize:
                        scores_arr = scores_arr * -1
                    pred = self.model(valid_zs)
                    loss = -self.mll(pred, scores_arr.cuda())
                    if self.train_c is not None:
                        for ix, c_model in enumerate(self.c_models):
                            pred2 = c_model(valid_zs.cuda())
                            loss += -self.c_mlls[ix](
                                pred2, constraints_tensor[:, ix].cuda()
                            )
                    optimizer1.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer1.step()
                    with torch.no_grad():
                        z = z.detach().cpu()
                        self.update_next(
                            z, scores_arr, xs_list, c_next_=constraints_tensor
                        )
            torch.cuda.empty_cache()
        self.model.eval()
        if self.train_c is not None:
            for c_model in self.c_models:
                c_model.eval()

        return self

    def acquisition(self):
        """Generate new candidate points,
        evaluate them, and update data
        """
        # 1. Generate a batch of candidates in
        #   trust region using surrogate model
        if self.train_c is not None:  # if constrained
            constraint_model_list = self.c_models
        else:
            constraint_model_list = None
        z_next = generate_batch(
            state=self.tr_state,
            model=self.model,
            X=self.train_z,
            Y=self.train_y,
            batch_size=self.bsz,
            acqf=self.acq_func,
            constraint_model_list=constraint_model_list,
        )
        if self.objective.similarity is not None:
            # We need to get the GP lengthscales
            ls = self.model.get_lengthscales()
            breakpoint()

        # 2. Evaluate the batch of candidates by calling oracle
        with torch.no_grad():
            out_dict = self.objective(z_next)
            z_next = out_dict["valid_zs"]
            y_next = out_dict["scores"]
            x_next = out_dict["decoded_xs"]
            c_next = out_dict["constr_vals"]
            if self.minimize:
                y_next = y_next * -1
        # 3. Add new evaluated points to dataset (update_next)
        if len(y_next) != 0:
            y_next = torch.from_numpy(y_next).float()
            self.update_next(z_next, y_next, x_next, c_next, acquisition=True)
        else:
            self.progress_fails_since_last_e2e += 1
            if self.verbose:
                print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")

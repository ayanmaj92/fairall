"""
Authors: Ayan Majumdar, Miriam Rateike
"""

import torch
from torch.nn.functional import sigmoid
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.dense import MLPModule
from utils.constants import Cte
from functools import partial
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import random

# FZ_TRAIN_LABEL_ONLY_UNLABELED = False


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_loss(u_true, u_prob_pred, ips_prob, cost):
    u_true = u_true.unsqueeze(dim=-1)
    p_wgt = torch.tensor((1 - cost) / cost)
    loss = F.binary_cross_entropy_with_logits(u_prob_pred, u_true, weight=(cost / ips_prob),
                                              reduction="mean", pos_weight=p_wgt)
    return loss


class FzNet(nn.Module):
    def __init__(self, clf_list, act, drop_rate, model_type, seed):
        """
        Neural network for FZ policy (Policy model that takes Z as input)
        @param clf_list: hidden layers' list
        @param act: activation function
        @param drop_rate: dropout rate
        @param model_type: NN model or LR (logistic regression)
        @param seed: random seed
        """
        super().__init__()
        set_seed(seed)
        if model_type == Cte.MODEL_LR:
            self.model = nn.Sequential(
                nn.Linear(clf_list[0], 1),
            )
        else:
            self.model = MLPModule(h_dim_list=clf_list,
                                   activ_name=act,
                                   bn=False,
                                   drop_rate=drop_rate,
                                   net_type='clf')

        def init_weights(m, gain=1.):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0.01)
        self.model.apply(partial(init_weights, gain=nn.init.calculate_gain('relu')))

    def forward(self, x):
        return self.model(x)


class Policy:
    def __init__(self):
        self.thresh = None
        self.pol_sampler = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def get_new_mask(self, old_mask, prob_u):
        if self.pol_sampler == 'DET':
            pred_u = (prob_u >= self.thresh).long().flatten()
        elif self.pol_sampler == 'LOG':
            pred_u = torch.bernoulli(prob_u).long().flatten()
        else:
            raise NotImplementedError
        new_mask = torch.zeros_like(old_mask, dtype=torch.bool)
        # Apply predicted u to change the mask.
        new_mask[pred_u == 1, :] = torch.ones(new_mask.size(1), dtype=torch.bool)
        return new_mask


class PolicyQxs(Policy):
    def __init__(self, config):
        """
        Policy class for classifier policy model
        """
        super().__init__()
        self.seed = config["seed"]
        self.thresh = config["model"]["params"]["costs"]
        self.pol_sampler = config["model"]["params"]["pol_sampler"]

    @torch.no_grad()
    def __call__(self, model, data, old_mask):
        """
        :param model: SSCVAE model that will be used to generate policy.
        :param data: Data array
        :param old_mask: Mask from data loader
        :return: new mask
        """
        model.eval()
        xu_ = data[:, :-1]
        s_ = data[:, -1]
        classifier_input = [xu_, s_]
        mask_unsup = torch.ones_like(data)
        prob_u = sigmoid(model.classify(*classifier_input, mask_unsup))
        set_seed(self.seed)
        new_mask = self.get_new_mask(old_mask, prob_u)
        return new_mask, prob_u

    def __str__(self):
        return "Policy U ~ Q(X,S)"


class PolicyPzs(Policy):
    def __init__(self, config):
        super().__init__()
        self.thresh = config["model"]["params"]["costs"]
        self.pol_sampler = config["model"]["params"]["pol_sampler"]
        self.seed = config["seed"]

    @torch.no_grad()
    def __call__(self, model, data, old_mask, is_sup=False):
        """
        :param model: SSCVAE model that will be used to generate policy.
        :param data: Data array
        :param old_mask: Mask from data loader
        :param is_sup: is supervised data flag
        :return: new mask
        """
        model.eval()
        # The data we need to apply our policy to is unsupervised.
        decoder_u, _, theta = model.phase_2_policy_helper(data, is_sup=is_sup)
        bern_p = sigmoid(theta[:, -1])
        bern_p = bern_p.reshape(-1, 1)

        set_seed(self.seed)
        new_mask = self.get_new_mask(old_mask, bern_p)
        return new_mask, bern_p

    def __str__(self):
        return "Policy U ~ P(z,S)"


class PolicyFz(Policy):
    def __init__(self, config, epochs=None):
        super().__init__()
        self.seed = config["seed"]
        cfg = config
        self.thresh = cfg["model"]["params"]["costs"]
        inp_size = cfg["model"]["params"]["latent_size"]
        h_dim = cfg["model"]["params"]["h_dim_list_clf"]
        act = cfg["model"]["params"]["act_name"]
        drop_rate = cfg["model"]["params"]["drop_rate_clf"]
        self.costs = cfg['model']['params']['costs']
        self.method = cfg['model']['params']['fz_method']
        self.pol_sampler = config["model"]["params"]["pol_sampler"]

        # Defining model
        likelihood_u_params_size = 1
        clf_list = [inp_size]
        clf_list.extend(h_dim)
        clf_list.append(likelihood_u_params_size)
        mod_type = cfg["model"]["params"]["model_type"]
        self.pol_model = FzNet(clf_list, act, drop_rate, mod_type, cfg["seed"])
        if epochs is None:
            self.epochs = cfg['trainer2']['epochs_per_ts']
        else:  # Will use this only during warmup, when we train this model AFTER training VAE.
            self.epochs = epochs
        self.batch_size = cfg["dataset"]["params2"]["batch_size"]
        self.pol_fitted = False
        self.qxs_helper = PolicyQxs(cfg)
        self.opt = Adam(self.pol_model.parameters(), lr=cfg['optimizer']['params']['learning_rate'])

    def train_policy(self, model, data, mask, prob1):
        """
        :param model: SSCVAE model that we need for training policy.
        :param data: data for training policy
        :param mask: mask that might be needed to filter labeled data
        :param prob1: IPS weights, will need in some cases
        :return: None (we only internally train the pol_model)
        Note: We can have different approaches here:
        a. Use both labeled and unlabeled, all labels clf output --> (CLF)
        b. (Ours, used in main paper) Use both labeled and unlabeled, all labels dec output --> (DEC)
        c. Use only labeled, then need IPS weights in loss! --> (LAB)
        """
        model.eval()
        set_seed(self.seed)
        # 1. Get z and u for training. Depends on which method we will use!!
        idx_sup = (mask[:, -1] == 1).nonzero(as_tuple=False)
        idx_unsup = (mask[:, -1] == 0).nonzero(as_tuple=False)

        if self.method in [Cte.FZ_DEC, Cte.FZ_CLF]:
            # Get labeled and unlabeled idx
            z_, u_ = torch.Tensor(), torch.Tensor()
            for j, idx in enumerate([idx_unsup, idx_sup]):
                if j == 0:
                    is_sup = False
                else:
                    is_sup = True
                if len(idx) > 0:
                    data_idx = torch.index_select(data, 0, idx.squeeze())
                    mask_idx = torch.index_select(mask, 0, idx.squeeze())
                    u_label_idx, z_idx, theta_idx = model.phase_2_policy_helper(data_idx, is_sup=is_sup)
                    bern_p = sigmoid(theta_idx[:, -1])
                    bern_p = bern_p.reshape(-1, 1)
                    u_label_idx = (bern_p >= self.thresh).long().flatten()
                    if self.method == Cte.FZ_CLF:
                        # Here we need to get classifier U
                        _, prob_u_idx = self.qxs_helper(model, data_idx, mask_idx)
                        u_label_idx = (prob_u_idx >= self.thresh).long().flatten()
                    # if is_sup and FZ_TRAIN_LABEL_ONLY_UNLABELED:
                    #     u_label_idx = data_idx[:, -2].long().flatten()
                    z_ = torch.cat([z_, z_idx], 0)
                    u_ = torch.cat([u_, u_label_idx], 0)
            rand_idx = torch.randperm(z_.size()[0])
            train_z = z_[rand_idx]
            u_label = u_[rand_idx]
            sample_wgt = None
        elif self.method in [Cte.FZ_LAB]:
            # method uses only labeled data!
            x_sup = torch.index_select(data, 0, idx_sup.squeeze())
            u_label = x_sup[:, -2]
            if idx_sup.shape[0] < 1:
                if self.pol_fitted is False:
                    self.pol_fitted = True
                return
            _, train_z, _ = model.phase_2_policy_helper(x_sup, is_sup=True)
            # Here as we use only labeled data, we would need the IPS weights.
            assert prob1 is not None, "If using only labeled data to train policy FZ, " \
                                      "we need IPS weights! prob1 cannot be None!"
            prob1_lab = torch.index_select(prob1, 0, idx_sup.squeeze())
            sample_wgt = prob1_lab
        else:
            raise NotImplementedError

        if sample_wgt is None:
            sample_wgt = torch.ones_like(prob1)
        # Detach to ensure that backward will work properly only for this network.
        train_z = train_z.detach()
        dataset_ = TensorDataset(train_z, u_label, sample_wgt)
        data_loader = DataLoader(dataset_, batch_size=len(dataset_) // 3 if len(dataset_) >= 3 else len(dataset_),
                                 shuffle=True, drop_last=False)

        with torch.enable_grad():
            self.pol_model.train()
            for epoch in range(self.epochs):
                loss_print = None
                for batch_num, batch in enumerate(data_loader):
                    x, y, prob1 = batch
                    self.opt.zero_grad()
                    out_logits = self.pol_model(x)
                    loss = compute_loss(y, out_logits, prob1, self.costs)
                    loss_print = loss
                    loss.backward()
                    self.opt.step()
                if loss_print is not None:
                    print("Training FZ || Epoch", epoch + 1, "|| Loss:", loss_print.item(), "||")
        self.pol_fitted = True

    @torch.no_grad()
    def __call__(self, model, data, old_mask):
        """
        :param model: SSCVAE model that will be used to generate policy
        :param data: Data array
        :param old_mask: Mask from data loader
        :return: new mask
        """
        model.eval()
        # Training pol_model
        # 1. Get z for new data.
        _, new_z, _ = model.phase_2_policy_helper(data, is_sup=False)
        # 2. Predict u using policy model.
        self.pol_model.eval()
        pred_proba_u = sigmoid(self.pol_model(new_z))
        set_seed(self.seed)
        new_mask = self.get_new_mask(old_mask, pred_proba_u)
        # Ensuring numerical stability
        pred_proba_u += 1e-8
        pred_proba_u = torch.tensor(pred_proba_u.reshape(-1, 1))
        # 4. Return
        return new_mask, pred_proba_u

    def __str__(self):
        return "Policy U ~ F(z) " + str(self.method)


class PolicyNA(Policy):
    # This policy is supposed to do nothing, simply return pi_0 stuff.
    def __init__(self):
        super().__init__()

    def __call__(self, old_mask, old_prob1):
        return old_mask, old_prob1

    def __str__(self):
        return "Policy U ~ Pi_0"

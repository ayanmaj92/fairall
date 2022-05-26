"""
Authors: Miriam Rateike, Ayan Majumdar
Based on an older version of https://github.com/adrianjav/heterogeneous_vaes
"""

from functools import partial
import torch
import torch.nn as nn
import torch.distributions as dists
from torch.nn.functional import softplus, sigmoid, binary_cross_entropy_with_logits
from torch.distributions import kl_divergence
from modules.dense import MLPModule
from models.cvae import CVAE
from utils.constants import Cte
import numpy as np
import random
from fairtorch import DemographicParityLoss


def set_seed(seed):
    """
    Set random seed
    @param seed: integer seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class SSCVAE(nn.Module):
    def __init__(self, prob_model, latent_size, h_dim_list_clf, h_dim_list_enc, h_dim_list_dec, act_name,
                 drop_rate_clf, conditional, alpha, costs, seed, kl_beta, lmbd,
                 loss_function, model_type, phase):
        """
        Initializes SSCVAE class object
        @param prob_model: probabilistic model object
        @param latent_size: latent size of VAE
        @param h_dim_list_clf: hidden layer sizes of classifier
        @param h_dim_list_enc: hidden layer sizes of encoder
        @param h_dim_list_dec: hidden layer sizes of decoder
        @param act_name: activation function to use
        @param drop_rate_clf: dropout rate for classifier
        @param conditional: whether to train conditional model w.r.t. sensitive S
        @param alpha: hyperparameter to weigh classifier loss
        @param costs: classification cost (0-1)
        @param seed: random seed to set
        @param kl_beta: term to weight KL loss term
        @param lmbd: term to weight Demographic Parity loss (for niki-fair/FairLog)
        @param loss_function: which methodology to train with
        @param model_type: whether to use logistic regression model or NN model for classifier
        @param phase: whether phase 1 or phase 2 training
        """
        super().__init__()

        if loss_function != Cte.LOSS_FAIRLOG:
            self.cvae = CVAE(prob_model, latent_size, h_dim_list_enc, h_dim_list_dec, act_name, conditional, kl_beta,
                             seed, phase)
            print("cvae", self.cvae)
        self.prob_model = prob_model
        self.num_params = prob_model.num_params
        self.input_scaler = prob_model.InputScaler
        self.costs = costs
        self.alpha = alpha
        self.dim_z = latent_size
        self.seed = seed
        self.af = kl_beta
        self.conditional = conditional
        self.loss_function = loss_function
        self.phase = phase
        self.lmbd = lmbd

        set_seed(seed)

        # Classifier
        if self.conditional:
            classifier_input_size = prob_model.domain_size  # because without u, i.e. x, s
        else:
            classifier_input_size = prob_model.domain_size - 1  # because without u, i.e. x only

        if (loss_function == Cte.LOSS_FAIRLOG) and (model_type == Cte.MODEL_LR):
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, 1),
            )
        else:
            likelihood_u_params_size = 1
            clf_list = [classifier_input_size]
            clf_list.extend(h_dim_list_clf)
            clf_list.append(likelihood_u_params_size)
            self.classifier = MLPModule(h_dim_list=clf_list,
                                        activ_name=act_name,
                                        bn=False,
                                        drop_rate=drop_rate_clf,
                                        net_type='clf')
            print("clf", self.classifier)

        def init_weights(m, gain=1.):
            """
            @param m: layer
            @param gain: gain value for initialization
            """
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0.01)

        self.classifier.apply(partial(init_weights, gain=nn.init.calculate_gain('relu')))

    def mc_kl(self, prob1, qz0, qz1, pz, K=100):
        """
        Compute Monte-Carlo estimate of KL loss
        @param prob1: policy probability values (of positive decision)
        @param qz0: posterior likelihood for positive prediction
        @param qz1: posterior likelihood for negative prediction
        @param pz: prior likelihood
        @param K: number of MC samples
        @return: KL value over MC samples
        """
        # sample
        z0_samples = qz0.rsample(sample_shape=torch.Size([K]))  # 100, 500, 2 (K, batch, dimz)
        z1_samples = qz1.rsample(sample_shape=torch.Size([K]))  # 100, 500, 2

        # resize
        _prob1 = prob1.unsqueeze(0)  # unsqueeze(0) 1, 500, 1

        # prior
        log_prob_pz_0 = pz.log_prob(z0_samples)
        log_prob_pz_1 = pz.log_prob(z1_samples)

        # integral
        qzu1_z0 = _prob1 * qz1.cdf(z0_samples)
        qzu0_z0 = (1 - _prob1) * qz0.cdf(z0_samples)
        qzu1_z1 = _prob1 * qz1.cdf(z1_samples)
        qzu0_z1 = (1 - _prob1) * qz0.cdf(z1_samples)

        log_sum_input_z0 = torch.cat((qzu0_z0.unsqueeze(-1), qzu1_z0.unsqueeze(-1)), 3)
        log_sum_input_z1 = torch.cat((qzu0_z1.unsqueeze(-1), qzu1_z1.unsqueeze(-1)), 3)

        log_prob_qz_0 = torch.logsumexp(log_sum_input_z0, 3)  # K x Batch x dim 2
        log_prob_qz_1 = torch.logsumexp(log_sum_input_z1, 3)  # K x Batch x dim 2

        kl0 = (1 - prob1) * (1 / K) * torch.sum((log_prob_qz_0) - (log_prob_pz_0), dim=0)  # batch x dim 2
        kl1 = prob1 * (1 / K) * torch.sum((log_prob_qz_1) - (log_prob_pz_1), dim=0)  # 1000, 2

        return kl0 + kl1  # batch size x dim 2

    def classify(self, xu, s=None, mask=None):
        """
        Forward pass over classifier model
        @param xu: concatenated features (x) and utility (u)
        @param s: sensitive feature
        @param mask: mask for data labeled/unlabeled
        @return: classifier logit
        """
        if mask is not None:
            mask_u = mask[:, -1]
            mask_new = self.broadcast_mask(xu, mask_u)
            # because the classifier does not use u
            mask_new[:, -1] = 0
        else:
            print("Mask NONE!")

        # data point missing set to zero, 1 if observed, 0 if not
        xu_scaled = self.input_scaler(xu if mask is None else xu * mask_new.double())

        if self.conditional:
            x_scaled = xu_scaled[:, :-1]
        else:
            raise NotImplementedError

        clf_input = torch.cat((x_scaled, s.unsqueeze(-1)), dim=-1)
        # all unsupervised samples are set clf_input = 0
        return self.classifier(clf_input)

    def broadcast_mask(self, x, mask):
        new_mask = []
        for i in range(x.shape[1]):
            new_mask.append(mask.unsqueeze(-1))

        return torch.cat(new_mask, dim=-1)

    def forward(self, x, state, mask=None, prob1=None):
        # we assume no missing data, i.e. just have one mask
        has_unsup, has_sup = False, False
        # initialize all
        elbo_unsup, kl_unsup, logprob0_unsup, logprob1_unsup, recx_unsup, \
        elbo_sup, kl_sup, recxu_sup, clf_loss, clf_acc, clf_err, discrimination_loss \
            = torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor(
            [0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor(
            [0]), torch.tensor([0]), torch.tensor([0])

        idx_sup = (mask[:, -1] == 1).nonzero(as_tuple=False)
        idx_unsup = (mask[:, -1] == 0).nonzero(as_tuple=False)
        Nu = idx_unsup.shape[0]
        Nl = idx_sup.shape[0]

        if len(idx_unsup) > 0:
            has_unsup = True
        if len(idx_sup) > 0:
            has_sup = True

        if has_sup:
            x_sup = torch.index_select(x, 0, idx_sup.squeeze())
            prob1 = torch.index_select(prob1, 0, idx_sup.squeeze())
            xus_sup = x_sup.clone()
            xu_sup = x_sup[:, :-1]
            s_sup = x_sup[:, -1]
            u_sup = x_sup[:, -2]

            if self.conditional:
                mask_sup = torch.ones_like(xu_sup)
            else:
                mask_sup = torch.ones_like(xus_sup)

        if has_unsup and self.loss_function in [Cte.LOSS_FAIRALL]:
            x_all_unsup = torch.index_select(x, 0, idx_unsup.squeeze())
            xus_unsup = x_all_unsup.clone()
            x_unsup = x_all_unsup[:, :-2]
            xu_unsup = x_all_unsup[:, :-1]
            s_unsup = x_all_unsup[:, -1]

            if self.conditional:
                mask_unsup = torch.ones_like(xu_unsup)
            else:
                mask_unsup = torch.ones_like(xus_unsup)

        assert Nu + Nl == x.shape[0], "unlabeled and labeled data points do not sum up"

        if has_sup:
            # 1 ---- classifier on labeled data only
            if self.conditional:
                classifier_input = [xu_sup, s_sup]
            else:
                classifier_input = [xus_sup]

            if self.loss_function in [Cte.LOSS_FAIRLOG, Cte.LOSS_FAIRALL]:  # only then we need the classifier
                logits_u = self.classify(*classifier_input, mask_sup)
                prob_u = sigmoid(logits_u)

                # --- for loss function -----
                _u = u_sup.unsqueeze(dim=-1)
                clf_loss = self.compute_clf_loss(_u, logits_u, prob1)
                # - just for evaluation of how good we classify
                # accuracy
                pred_u = prob_u.clone()
                pred_u[pred_u >= self.costs] = 1
                pred_u[pred_u < self.costs] = 0
                clf_acc = (1 - (pred_u - _u) ** 2).sum()
                if self.loss_function == Cte.LOSS_FAIRLOG:
                    xu_all = x[:, :-1]
                    s_all = x[:, -1]
                    clf_in_all = [xu_all, s_all]
                    dp_loss = DemographicParityLoss(sensitive_classes=[-1, 1], alpha=self.lmbd)
                    discrimination_loss = dp_loss(xu_all[:, :-1],
                                                  self.classify(*clf_in_all, torch.ones_like(xu_all)), s_all).to("cpu")

            if self.loss_function in [Cte.LOSS_FAIRALL, Cte.LOSS_FAIRLAB]:
                # 2 --- Supervised VAE loss
                if self.conditional:
                    encoder_input = classifier_input
                else:
                    encoder_input = classifier_input

                z_params = self.cvae.encode(*encoder_input, mask_sup)

                qz = self.cvae.q_z(*z_params)
                z = qz.rsample()

                if self.conditional:
                    decoder_input = [z, s_sup]
                else:
                    decoder_input = [z]

                # no mask, no scaling
                theta = self.cvae.decode(*decoder_input)  # batch_size x num_params
                # conditional is handled inside likelihood
                log_prob = self.cvae.my_log_likelihood(xus_sup, mask_sup, theta=theta)

                if self.loss_function == Cte.LOSS_FAIRALL:
                    recxu_sup = log_prob.sum()
                    kl_sup = torch.sum(kl_divergence(qz, self.cvae.prior_z))  # 500, 2

                elif self.loss_function == Cte.LOSS_FAIRLAB:
                    recxu_sup = (log_prob * prob1).sum()  # elementwise multiplication
                    kl_sup = torch.sum(kl_divergence(qz, self.cvae.prior_z) * prob1)

                # --- for loss function -----

                elbo_sup = recxu_sup - self.af * kl_sup

        if has_unsup and self.loss_function in [Cte.LOSS_FAIRALL]:
            # 3 --- Unupervised VAE loss
            if self.conditional:
                classifier_input = [xu_unsup, s_unsup]
            else:
                classifier_input = [xus_unsup]

            with torch.no_grad():
                prob_u = sigmoid(self.classify(*classifier_input, mask_unsup))

            pred_u = torch.bernoulli(prob_u).float()

            x_predu_unsup = torch.cat((x_unsup, pred_u), -1)

            if self.conditional:
                encoder_input = [x_predu_unsup, s_unsup]
            else:
                encoder_input = [x_predu_unsup]

            z01_params = self.encode(*encoder_input, mask_unsup)  # [1, 0, 1] # returns z1_params, z0_params

            qz0, qz1 = self.q_z(*z01_params)

            L = 50  # How many samples?
            # 1. Sampling L samples. returns LxBxZ
            z0 = qz0.rsample(torch.Size([L]))
            z1 = qz1.rsample(torch.Size([L]))
            # get BxLxZ
            z0, z1 = z0.transpose(0, 1), z1.transpose(0, 1)
            # 2. Flattening 1st 2 dims. Should be {B*L=M}xZ
            z0 = z0.reshape(-1, z0.size(-1))
            z1 = z1.reshape(-1, z1.size(-1))
            # 3. Repeat s, then append it.
            rep_s_unsup = torch.repeat_interleave(s_unsup, L)
            if self.conditional:
                decoder_input0 = [z0.squeeze(), rep_s_unsup]
                decoder_input1 = [z1.squeeze(), rep_s_unsup]
            else:
                decoder_input1 = [z1]
                decoder_input0 = [z0]
            # 4. Pass through decoder.
            theta0 = self.cvae.decode(*decoder_input0)  # M x num_params
            theta1 = self.cvae.decode(*decoder_input1)  # M x num_params
            # 5. Repeat data, mask. Compute log-prob.
            rep_xus_unsup = xus_unsup.repeat_interleave(L, 0)
            rep_mask_unsup = mask_unsup.repeat_interleave(L, 0)
            log_prob0 = self.cvae.my_log_likelihood(rep_xus_unsup, rep_mask_unsup, theta=theta0)
            log_prob1 = self.cvae.my_log_likelihood(rep_xus_unsup, rep_mask_unsup, theta=theta1)
            # select first two columns, not u
            log_prob0 = log_prob0[:, :-1]  # sum first rows, then columns, but not u
            log_prob1 = log_prob1[:, :-1]  # sum, but not u
            # 6. KL_unsup.
            # you can pass here as last argument K, otherwise it takes by default K=100
            # we also sum above the kl_divergence like this
            kl_z = torch.sum(self.mc_kl(prob_u, qz0, qz1, self.cvae.prior_z))  # sum over batch
            rep_kl_z = kl_z.repeat_interleave(L)
            # 7. Compute elbo_unsup
            rep_prob_u = prob_u.repeat_interleave(L, 0)
            recx_unsup = ((1 - rep_prob_u) * log_prob0 + rep_prob_u * log_prob1).sum()
            kl_unsup = torch.sum(rep_kl_z)

            elbo_unsup = (recx_unsup - (self.af * kl_unsup))  # kl sum over L.

            logprob0_unsup, logprob1_unsup = log_prob0.sum(), log_prob1.sum()

            elbo_unsup = elbo_unsup / L
            recx_unsup = recx_unsup / L
            kl_unsup = kl_unsup / L
            logprob0_unsup = logprob0_unsup / L
            logprob1_unsup = logprob1_unsup / L

        sup_elb, sup_clf = torch.tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
        unsup_elb = torch.tensor([0.0], requires_grad=True)

        if has_sup and self.loss_function:
            if self.loss_function in [Cte.LOSS_FAIRALL, Cte.LOSS_FAIRLAB]:
                sup_elb = -(elbo_sup / Nl)

            if self.loss_function in [Cte.LOSS_FAIRALL, Cte.LOSS_FAIRLOG]:
                sup_clf = self.alpha * clf_loss / Nl
                if self.loss_function == Cte.LOSS_FAIRLOG:
                    # Here we are adding the discrimination loss, ONLY if we are doing Niki/FairLog.
                    #  The formula is alpha * (clf_loss + lmbd * discrimination)
                    sup_clf += self.alpha * self.lmbd * discrimination_loss

        if has_unsup and self.loss_function in [Cte.LOSS_FAIRALL]:
            unsup_elb = -(elbo_unsup / Nu)

        total_loss = sup_elb + sup_clf + unsup_elb

        if hasattr(state, 'metrics'):
            with torch.no_grad():
                state.metrics.update({'total': total_loss.item()})
                if has_unsup:
                    state.metrics.update({
                        '-elbo_unsup': -elbo_unsup.item() / Nu, 'kl_unsup': kl_unsup.item() / Nu,
                        '-logprob0_unsup': -logprob0_unsup.item() / Nu, '-logprob1_unsup': -logprob1_unsup.item() / Nu,
                        '-recx_unsup': -recx_unsup.item() / Nu
                    })
                if has_sup:
                    state.metrics.update({
                        '-elbo_sup': -elbo_sup.item() / Nl, 'kl_sup': kl_sup.item() / Nl,
                        '-logprob_sup': -recxu_sup.item() / Nl,
                        'clf_acc': clf_acc.item() / Nl, 'clf_loss': clf_loss.item() / Nl,
                        'discr_loss': discrimination_loss.item()
                    })
        return total_loss

    @torch.no_grad()
    def reconstruct(self, x, mask):
        """
        Method to reconstruct data
        @param x: input features
        @param mask: input mask
        @return: CVAE reconstructed data.
        """
        set_seed(self.seed)
        return self.cvae.reconstruct(x, mask)

    def log_likelihood(self, x, state, mask, theta=None):
        return self.cvae.log_likelihood(x, state, mask, theta)

    def compute_clf_loss(self, u_true, u_prob_pred, ips_prob):
        if ips_prob is not None:
            p_wgt = torch.tensor((1 - self.costs) / self.costs)
            loss = binary_cross_entropy_with_logits(u_prob_pred, u_true, weight=(self.costs / ips_prob),
                                                    reduction="sum", pos_weight=p_wgt)
            return loss
        else:
            print("Oh no, prob1 is None, cannot de-bias with IPS!")
            raise NotImplementedError

    def encode(self, x, s=None, mask=None):
        """
        Method for encoder forward pass
        @param x: non-sensitive data features
        @param s: sensitive feature
        @param mask: mask for labeled/unlabeled
        @return: encoder output of mean/variance of latent
        """
        mask_u = mask[:, -1]
        mask_new = self.broadcast_mask(x, mask_u)
        # because the classifier does not use u
        mask_new[:, -1] = 0
        # (1) scale and return x,s with mask == False
        x_scaled = self.input_scaler(
            x if mask is None else x * mask_new.double())  # data point missing set to zero, 1if observed, 0 if not
        # (2) remove u
        x_scaled = x_scaled[:, :-1]
        # (3) add to x_scale u=1 and u=0
        x_scaled0 = torch.cat((x_scaled, torch.zeros(x_scaled.shape[0]).unsqueeze(-1)), -1)
        x_scaled1 = torch.cat((x_scaled, torch.ones(x_scaled.shape[0]).unsqueeze(-1)), -1)
        # (4) send through encoder both cases
        if self.conditional:
            h0 = self.cvae.encoder(torch.cat((x_scaled0, s.unsqueeze(-1)), dim=-1))  # in [x1, x2, u=0, s]
            h1 = self.cvae.encoder(torch.cat((x_scaled1, s.unsqueeze(-1)), dim=-1))  # in [x1, x2, u=1, s]
        else:
            h0 = self.cvae.encoder(x_scaled0)  # in [x1, x2, u=0, s]
            h1 = self.cvae.encoder(x_scaled1)  # in [x1, x2, u=1, s]
        # (5) compute for both cases
        loc0 = self.cvae.encoder_loc(h0)  # constraints.real
        log_scale0 = self.cvae.encoder_logscale(h0)  # constraints.real
        loc1 = self.cvae.encoder_loc(h1)  # constraints.real
        log_scale1 = self.cvae.encoder_logscale(h1)  # constraints.real
        # (6) return all six variables (2 for each)
        return loc0, log_scale0, loc1, log_scale1

    def decode(self, z, s=None):
        """
        Method for decoder forward pass
        @param z: latent vector
        @param s: sensitive feature for conditional
        @return: decoder output likelihoods
        """
        if self.conditional:
            return self.decoder(torch.cat((z, s.unsqueeze(-1)), dim=-1))
        else:
            return self.decoder(z)

    def q_z(self, loc0, log_scale0, loc1, log_scale1):
        scale0 = softplus(log_scale0)
        z0 = dists.Normal(loc0, scale0)
        scale1 = softplus(log_scale1)
        z1 = dists.Normal(loc1, scale1)
        return z0, z1

    def q_z_joint(self, loc0, log_scale0, loc1, log_scale1, prob_u):
        scale0 = softplus(log_scale0)
        scale1 = softplus(log_scale1)
        z01 = dists.Normal((1 - prob_u) * loc0 + prob_u * loc1, (1 - prob_u) * scale0 + prob_u * scale1)
        return z01

    def phase_2_policy_helper(self, x, is_sup=False):
        # New method for use in Phase 2 policies.
        # This code is following the forward function... Inspired from supervised and unsupervised.
        set_seed(self.seed)
        xus_ = x.clone()
        x_ = x[:, :-2]
        xu_ = x[:, :-1]
        s_ = x[:, -1]

        # 1. Generate mask because latter methods use it.
        if self.conditional:
            mask_ = torch.ones_like(xu_)
        else:
            mask_ = torch.ones_like(xus_)
        # 2. Use classifier to get u to be fed to encoder.
        if self.conditional:
            classifier_input = [xu_, s_]
        else:
            classifier_input = [xus_]
        prob_u = sigmoid(self.classify(*classifier_input, mask_))
        u_pred = torch.bernoulli(prob_u).float()
        # 3. Use encoder to get z.
        if not is_sup:
            # a. If we are using for unsupervised data (to get decoded z,s->u policy)
            x_probu_ = torch.cat((x_, u_pred), -1)
            if self.conditional:
                encoder_input = [x_probu_, s_]
            else:
                encoder_input = [x_probu_]
            z_params = self.cvae.encode(*encoder_input, mask_)
        else:
            # b. If we are using for supervised data (to train z->u policy)
            encoder_input = classifier_input
            z_params = self.cvae.encode(*encoder_input, mask_)
        qz = self.cvae.q_z(*z_params)
        z = qz.rsample()
        # 4. Use decoder like a supervised model to get reconstructed u.
        if self.conditional:
            decoder_input = [z, s_]
        else:
            decoder_input = [z]
        # no mask, no scaling
        theta = self.cvae.decode(*decoder_input)  # batch_size x num_params
        # 5. Reconstruct data like generate_data() method.
        rec_data = self.prob_model(*theta.unbind(dim=-1)).sample()
        # 6. Return decoder U and latent z.
        return rec_data[:, -1], z, theta

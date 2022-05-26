"""
Authors: Miriam Rateike, Ayan Majumdar
Based on an older version of https://github.com/adrianjav/heterogeneous_vaes
"""

from functools import partial
import torch
import torch.nn as nn
import torch.distributions as dists
from torch.nn.functional import softplus
from torch.distributions import kl_divergence
from modules.dense import MLPModule
import numpy as np
import random


class CVAE(nn.Module):
    """Class definition of CVAE model"""
    def __init__(self, prob_model, latent_size, h_dim_list_enc,
                 h_dim_list_dec, act_name, conditional,
                 kl_beta, seed, phase):
        """
        prob_model: probabilistic model class object
        latent_size: size of the latent dimension
        h_dim_list_enc: list of hidden layer sizes of encoder network
        h_dim_list_dec: list of hidden layer sizes of decoder network
        act_name: activation used in networks
        conditional: whether to do a conditional on sensitive feature
        kl_beta: set the value of beta for KL weighting in VAE loss
        seed: set the seed value
        phase: whether CVAE model is being used in phase 1 or 2
        """
        super().__init__()
        self.phase = phase
        self.prob_model = prob_model
        self.num_params = prob_model.num_params
        self.input_scaler = prob_model.InputScaler
        self.kl_beta = kl_beta
        self.latent_size = latent_size
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Prior
        self.register_buffer('prior_z_loc', torch.zeros(latent_size))
        self.register_buffer('prior_z_scale', torch.ones(latent_size))
        self.conditional = conditional

        # Encoder
        if self.conditional:
            encoder_input_size = prob_model.domain_size + 1
        else:
            encoder_input_size = prob_model.domain_size

        enc_list = [encoder_input_size]
        enc_list.extend(h_dim_list_enc)

        self.encoder = MLPModule(h_dim_list=enc_list,
                                 activ_name=act_name,
                                 bn=False,
                                 drop_rate=0.0,  # no dropout in VAE models
                                 net_type='enc')

        self.encoder_loc = nn.Linear(enc_list[-1], latent_size)
        self.encoder_logscale = nn.Linear(enc_list[-1], latent_size)

        # Decoder
        if self.conditional:
            decoder_input_size = latent_size + 1
        else:
            decoder_input_size = latent_size

        dec_list = [decoder_input_size]
        dec_list.extend(h_dim_list_dec)
        dec_list.append(self.num_params)
        self.decoder = MLPModule(h_dim_list=dec_list,
                                 activ_name=act_name,
                                 bn=False,
                                 drop_rate=0.0,  # no dropout in VAE models
                                 net_type='dec')

        def init_weights(m, gain=1.):
            """
            @param m: layer
            @param gain: gain value for initialization
            """
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0.01)

        self.encoder.apply(partial(init_weights, gain=nn.init.calculate_gain('relu')))
        self.encoder_loc.apply(init_weights)
        self.encoder_logscale.apply(init_weights)
        self.decoder.apply(partial(init_weights, gain=nn.init.calculate_gain('relu')))

    @property
    def prior_z(self):
        return dists.Normal(self.prior_z_loc, self.prior_z_scale)

    def q_z(self, loc, logscale):
        scale = softplus(logscale)
        return dists.Normal(loc, scale)

    def encode(self, x, s=None, mask=None):
        """
        Method for encoder
        @param x: non-sensitive features
        @param s: sensitive feature
        @param mask: mask for data labeled/unlabeled
        @return: latent variable loc (mean) and logscale (log of variance)
        """
        # Normal distribution
        x_scaled = self.input_scaler(
            x if mask is None else x * mask.double())  # data point missingness, 1 if observed, 0 if not
        if self.conditional:
            h = self.encoder(torch.cat((x_scaled, s.unsqueeze(-1)), dim=-1))
        else:
            h = self.encoder(x_scaled)

        loc = self.encoder_loc(h)
        logscale = self.encoder_logscale(h)

        return loc, logscale

    def decode(self, z, s=None):
        """
        Method for decoder
        @param z: latent variable after reparameterization
        @param s: sensitive feature for conditional
        @return: output of decoder network; distributions
        """
        if self.conditional:
            # Adding this small code to ensure that z is never 1-dimensional.
            #  Can happen if we get single data-point as input.
            if self.latent_size > 1 and len(z.size()) < 2:
                z = z.unsqueeze(0)
            elif self.latent_size == 1 and len(z.size()) < 2:
                z = z.unsqueeze(-1)
            return self.decoder(torch.cat((z, s.unsqueeze(-1)), -1))
        else:
            return self.decoder(z)

    def forward(self, x, state, mask=None):
        """
        Forward pass of CVAE model
        @param x: non-sensitive data features
        @param state: Ignite engine state object
        @param mask: mask for data labeled/unlabeled
        @return: elbo loss
        """
        N = x.size()[0]  # needed for likelihood computation

        if self.conditional:
            xu = x[:, :-1]
            s = x[:, -1]
            encoder_input = [xu, s]
        else:
            encoder_input = [x]

        z_params = self.encode(*encoder_input, mask)

        z = self.q_z(*z_params).rsample()

        # input here z, s
        if self.conditional:
            decoder_input = [z, s]
        else:
            decoder_input = [z]

        theta = self.decode(*decoder_input)  # batch_size x num_params

        # conditional is handled inside likelihood
        log_prob = self.my_log_likelihood(x, mask, theta=theta)
        ll = log_prob.clone().sum(dim=0).sum(dim=0)
        kl_z = kl_divergence(self.q_z(*z_params), self.prior_z).sum()
        elbo = ll - self.kl_beta * kl_z

        # when writing losses it will be averaged
        if hasattr(state, 'metrics'):
            with torch.no_grad():
                self.prob_model.eval()
                state.metrics.update({
                    '-elbo': -elbo.item() / N, 'kl_z': kl_z.item() / N
                })
                # conditional is handled inside likelihood
                log_prob = self.log_likelihood(x, state, None, theta=theta).mean(dim=0)
                state.metrics.update({'-re': -log_prob.sum() / N})
                state.metrics.update({f'-re_{i}': -l_i.item() / N
                                      for i, l_i in enumerate(log_prob)})
                self.prob_model.train()

        return -elbo

    @torch.no_grad()
    def reconstruct(self, x, mask):
        """
        Function to reconstruct input data.
        @param x: non-sensitive input features
        @param mask: mask for data labeled/unlabeled
        @return: probabilistic model estimate of reconstructed features
        """
        # feed to encoder
        if self.conditional:
            s = x[:, -1]
            x = x[:, :-1]

            encoder_input = [x, s]
        else:
            encoder_input = [x]

        # get latent z
        z_params = self.encode(*encoder_input, mask)
        z = self.q_z(*z_params).sample()

        # input here z, s to decoder
        if self.conditional:
            decoder_input = [z, s]
        else:
            decoder_input = [z]

        theta = self.decode(*decoder_input)  # batch_size x num_params

        if self.phase == 1:
            return self.prob_model(*theta.unbind(dim=-1)).sample(), z, None
        elif self.phase == 2:
            return self.prob_model(*theta.unbind(dim=-1)).sample(), z, self.prob_model(*theta.unbind(dim=-1))[-1].probs

    # Measures
    def my_log_likelihood(self, x, mask, theta=None):
        """
        Computes log-likelihood from just features. Used in forward pass.
        @param x: data features
        @param mask: mask for data labeled/unlabeled
        @param theta: probabilistic model parameters
        @return: log-likelihood value
        """
        if self.conditional:
            x = x[:, :-1]

        if mask is not None:
            assert x.shape[1] == mask.shape[1], "mask and x vector need to have the same number of columns"
        assert len(self.prob_model) == x.shape[1], "x needs to have the same size as the prob_model"

        log_prob = self.prob_model(*theta.unbind(dim=-1)).log_prob(x)  # batch_size x num_dimensions

        if mask is not None:
            log_prob = log_prob * mask.double()

        return log_prob

    def log_likelihood(self, x, state, mask, theta=None):
        """
        Computes log-likelihood from input features. Used for logging.
        @param x: data features
        @param state: Ignite state
        @param mask: mask for data labeled/unlabeled
        @param theta: parameters of probabilistic model
        @return: log likelihood computed value
        """
        if self.conditional:
            s = x[:, -1]
            x = x[:, :-1]

        if mask is not None:
            assert x.shape[1] == mask.shape[1], "mask and x vector need to have the same number of columns"
        assert len(self.prob_model) == x.shape[1], "x needs to have the same size as the prob_model"

        if theta is None:  # theta is parameters, outputs of parameters of likelihood
            if self.conditional:
                encoder_input = [x, s]
            else:
                encoder_input = [x]

            z_params = self.encode(*encoder_input, mask)
            if state is None:
                z = self.q_z(*z_params).sample()
            else:
                z = self.q_z(*z_params).rsample()

            # input here z, s
            if self.conditional:
                decoder_input = [z, s]
            else:
                decoder_input = [z]

            theta = self.decode(*decoder_input)  # batch_size x num_params

        log_prob = self.prob_model(*theta.unbind(dim=-1)).log_prob(x)  # batch_size x num_dimensions

        if mask is not None:
            log_prob = log_prob * mask.double()

        return log_prob

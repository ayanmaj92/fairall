"""
Authors: Ayan Majumdar, Miriam Rateike
Based on an older version of https://github.com/adrianjav/heterogeneous_vaes
"""

import random
import numpy as np
import torch
import torch.distributions
from torch.distributions import constraints
from sklearn.preprocessing import StandardScaler
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from utils.constants import Cte
import pandas as pd


def to_one_hot(x, size):
    x_one_hot = x.new_zeros(x.size(0), size)
    x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()
    return x_one_hot


def get_dataloader(cfg, prob_model,  phase, test, val, warmup, cf=False):
    """
    Helper function to get suitable DataLoader.
    @param cfg: config dictionary
    @param prob_model: probabilistic model object
    @param phase: phase 1 or 2
    @param test: true or false (data-loader for test data)
    @param val: true or false (data-loader for validation data)
    @param warmup: true or false (data-loader for warmup phase)
    @param cf: true or false (loading counterfactual data or not)
    @return: dataloader object
    """
    dataset = RealWorldDataset(path=f'datasets/{cfg["dataset"]["name"]}',
                               prob_model=prob_model,
                               latent_size=cfg["model"]["params"]['latent_size'],
                               conditional=cfg["model"]["params"]['conditional'],
                               file_type=cfg['file_type'],
                               categoricals=cfg['probabilistic']["categoricals"], test=test, semisup=cfg["semisup"],
                               policy=cfg["dataset"]["params2"]["init_policy"], seed=cfg["seed"],
                               percent=cfg["dataset"]["params2"]["percent"], val=val,
                               phase=phase,
                               warmup=warmup,
                               warmup_samples=cfg['trainer2']['warmup_samples'],
                               cf=cf,
                               phase1_samples=cfg['trainer1']['phase1_samples'])

    shuffle = not (test or val)  # shuffle, if not test or valid
    if phase == 2:
        # In Phase 2 ensuring we have enough batches to do stochastic learning and batches are not too large!
        batch_size = cfg["trainer2"]["samples_per_ts"] // 3
    else:
        if warmup:
            batch_size = cfg["trainer2"]["samples_per_ts"] // 3
        else:
            batch_size = cfg["dataset"]["params2"]["batch_size"]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader


def read_data_file(filename, categoricals, prob_model, conditional, semisup,
                   policy, percent, val_or_test, phase,
                   warmup, warmup_samples, phase1_samples):
    """
    Helper function to read the data file
    @param filename: name of file
    @param categoricals: which attributes are categorical
    @param prob_model: probabilistic model object
    @param conditional: do we condition?
    @param semisup: are we doing semi-supervised learning
    @param policy: which initial policy
    @param percent: what is percentage of data labeled (for some init policies)
    @param val_or_test: are we reading a validation or test data file?
    @param phase: which phase of training, 1 or 2
    @param warmup: are we loading data for warmup?
    @param warmup_samples: how many samples to use for warmup
    @param phase1_samples: how many samples for phase 1
    @return: data (torch tensor); init policy probability (torch tensor); Y_fair labels (torch tensor)
    """
    result = []
    df = pd.read_csv(filename, ',')
    if warmup:
        # take first X samples from the datafile
        df = df.iloc[:warmup_samples].copy()
    if phase == 1 and not val_or_test:
        if phase1_samples is not None:
            df = df.iloc[:phase1_samples].copy()

    if semisup and not val_or_test and warmup:
        if policy == "FUL":
            prob1 = pd.DataFrame(np.expand_dims(np.ones(len(df)), axis=1))
        elif policy == Cte.RAN:
            prob1 = pd.DataFrame(np.expand_dims(percent*np.ones(len(df)), axis=1))
        elif policy == "NO":
            prob1 = pd.DataFrame(np.expand_dims(np.zeros(len(df)), axis=1))
        else:
            prob1 = pd.DataFrame(df[policy])
    else:
        prob1 = None
    # remove all policies from dataset

    y = None
    if semisup:
        y = df[['Y']]
    if phase == 1 and warmup == True: # this is then only in warmup phase
        for i in list(df.columns):
            if i in [Cte.LENI, Cte.HARSH, Cte.LENIC, Cte.HARSHC, Cte.RAN]:
                # get decisions
                df = df.drop(i, axis=1)
                df = df.drop(f'{i}_D', axis=1)

    # Returning Y_F from here to set in Dataset class.
    # This would allow computing utility w.r.t. Y_F instead of Y.
    if semisup and 'Y_fair' in list(df.columns):
        dat_yf = torch.tensor(df[['Y_fair']].to_numpy())
    else:
        dat_yf = None

    # remove all other columns from dataset
    s_visited = False
    for i in list(df.columns):
        if s_visited:
            df = df.drop([i], axis=1)
        elif not semisup and i == 'Y':
            df = df.drop([i], axis=1)
        elif i == 'S':
            s_visited = True

    if semisup:
        assert df.columns[-1] == 'S' and df.columns[-2] == 'Y', \
            "Our assumption that the last column is S and second last U is not met"
    else:
        assert df.columns[-1] == 'S', \
            "Our assumption that the last column is S is not met"

    for i in categoricals:  # U is categorical
        df.iloc[:, i] -= df.iloc[:, i].min()

    for i, col in enumerate(df):
        # i is column name
        df[col] = df[col].fillna(df[col].max())
        if (conditional and (col is not 'S')) or (not conditional):  # one less prob_model
            if prob_model[i].__str__() == 'normal':
                scaler = StandardScaler()
                df.iloc[:, i] = scaler.fit_transform(df.iloc[:, i].to_numpy().reshape(-1, 1)).flatten()
            if prob_model[i].dist.support == constraints.positive:
                df[col] = df[col].astype('float64').clip(lower=1e-30)  # ensure that is positive

    assert len(pd.unique(df['S'])) == 2, "S needs to be binary"
    if semisup:
        assert len(pd.unique(df['Y'])) == 2, "Y needs to be binary"
    # convert to torch
    for _, line in df.iterrows():
        v = torch.tensor(line.tolist(), dtype=torch.float64)
        result += [v]

    data = torch.stack(result, dim=0)

    return data, prob1, dat_yf


class RealWorldDataset(Dataset, torch.nn.Module):
    def __init__(self, path, prob_model, categoricals, conditional,
                 latent_size, file_type, test, semisup, policy, seed,
                 percent, val, phase, warmup, warmup_samples, cf, phase1_samples):
        """
        @param path: path to data files
        @param prob_model: probabilistic model object
        @param categoricals: categorical features
        @param conditional: are we conditioning?
        @param latent_size: VAE latent size
        @param file_type: what is file extension?
        @param test: is this test data?
        @param semisup: are we doing semi-supervised learning?
        @param policy: which init policy?
        @param seed: random seed (int)
        @param percent: what percent of data to accept (for some init policy)
        @param val: is this validation data
        @param phase: which phase of training? 1 or 2
        @param warmup: is this warmup?
        @param warmup_samples: how many warmup samples?
        @param cf: is this counterfactual data?
        @param phase1_samples: how many phase 1 samples?
        """
        super().__init__()

        self.semisup = semisup
        self.phase = phase

        self.val = val
        self.test = test
        self.path = path

        if not test and not val:
            inp_file = f'{path}/data' + str(phase) + '.' + file_type
        elif test and not val:
            if not cf:
                inp_file = f'{path}/test' + str(phase) + '.' + file_type
            else:
                inp_file = f'{path}/cf_test' + str(phase) + '.' + file_type
        else:
            inp_file = f'{path}/valid' + str(phase) + '.' + file_type

        self.data, self.prob1, self.yf = \
            read_data_file(inp_file, categoricals, prob_model, conditional, semisup, policy,
                           percent, (test or val), phase, warmup, warmup_samples, phase1_samples)

        self.ncols = len(prob_model)
        self.latent_size = latent_size

        if semisup and (test or val):
            mask = torch.zeros(self.data.size(0))
        elif semisup and warmup and not (test or val):
            if policy in [Cte.LENI, Cte.HARSH, Cte.LENIC, Cte.HARSHC]:
                # Read the initial policy dependent decision.
                # This is mask, 1 is labeled, 0 is unlabeled.
                mask = self.read_mask(inp_file, policy, self.prob1)
            else:
                mask = self.create_mask(self.prob1, seed)
        else:
            mask = torch.ones(self.data.size(0))

        if conditional:
            dim_mask = self.data.shape[1] - 1
        else:
            dim_mask = self.data.shape[1]

        # synthetic
        nan_mask = torch.ones_like(mask, dtype=torch.float64)
        self.mask = (mask.long() + nan_mask.long()) == 2
        self.missing_mask = ((1 - mask.long()) + nan_mask.long()) == 2

        for i in range(dim_mask):
            if i == 0:
                new_mask = self.mask.unsqueeze(-1)
            else:
                new_mask = torch.cat((new_mask, self.mask.unsqueeze(-1)), dim=1)
        self.mask = new_mask

        if latent_size is not None:
            self.params = Parameter(torch.empty((len(self.data), latent_size)).uniform_(-1, 1))

        self.sens = None
        if conditional:
            self.sens = self.data[:, -1].clone()

        if semisup and not (test or val) and warmup:
            self.prob1 = torch.tensor(self.prob1.values)
        else:
            self.prob1 = torch.zeros(self.data.size(0))

    @property
    @torch.no_grad()
    def local_params(self):
        return self.params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.semisup and not (self.test or self.val):
            return self.data[idx], [self.params[idx], self.mask[idx], self.prob1[idx]]
        else:
            return self.data[idx], [self.params[idx], self.mask[idx]]

    def __str__(self):
        return f'Dataset: {self.path}'

    def create_mask(self, prob1, seed):
        """
        Method to create a mask on the fly, only for random initial policy.
        @param prob1: (float) probability to select an individual
        @param seed: (int) random seed
        @return: tensor mask
        """
        assert prob1 is not None, "probability cannot be None"
        random.seed(seed)
        mask = []
        for row in range(len(prob1)):
            mask_i = np.random.binomial(1, prob1.iloc[row].item(), 1).item()
            mask +=[mask_i]
        return torch.tensor(mask, dtype=torch.int)

    def read_mask(self, filename, policy, prob1):
        """
        Method to read CSV file and generate mask according to policy
        @param filename: CSV file to read
        @param policy: Selected initial policy
        @param prob1: Fraction of individuals to select
        @return: tensor mask
        """
        df = pd.read_csv(filename, ',')
        mask = df[f'{policy}_D'].iloc[:len(prob1)]
        print("read mask")
        return torch.tensor(mask.to_list(), dtype=torch.int)

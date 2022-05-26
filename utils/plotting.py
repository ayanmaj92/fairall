"""
Authors: Miriam Rateike, Ayan Majumdar
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution(data, discrete, **kwargs):
    """
    Plots the values according to whether the distribution is discrete or continuous (1-dimensional)
    """
    if discrete:
        if not isinstance(data, torch.Tensor):
            weights = np.ones_like(data[0]) / float(len(data[0]))
            weights = [weights] * len(data)
        else:
            weights = np.ones_like(data) / float(len(data))
        plt.hist([d.tolist() for d in data], alpha=0.5, weights=weights, **kwargs)  # bins=data[0].unique(), **kwargs)
    else:
        if 'color' in kwargs.keys():
            colors = kwargs.pop('color')
        else:
            colors = [None] * len(data)

        for d, color in zip(data, colors):
            plt.hist(d.tolist(), bins=100, alpha=0.5, density=True, color=color, **kwargs)


def plot_together(all_data, prob_model, title, path, dims=None, legend=None, **kwargs):
    prob_model.eval()
    colors = ['r', 'b', 'g']

    if dims is None:
        dims = range(len(prob_model))

    pos = 0
    for i, d in enumerate(prob_model):
        if str(d) == 'bernoulli*':
            # not using lipschitz scaling, so we should ideally never be here.
            d.is_discrete = True
        if str(d) == 'lognormal':
            plot_distribution([torch.log(d[..., pos]) for d in all_data], prob_model[pos].is_discrete,
                              color=colors[:len(all_data)], **kwargs)
        elif 'categorical' in str(d):
            plot_distribution([torch.argmax(data[..., pos: pos + d.domain_size], dim=-1) for data in all_data],
                              d.is_discrete, color=colors[:len(all_data)], **kwargs)
        else:
            plot_distribution([d[..., pos] for d in all_data], d.is_discrete, color=colors[:len(all_data)], **kwargs)

        plt.suptitle(title)
        if legend:
            plt.legend(legend)

        plt.savefig(f'{path}_{i}' if len(dims) > 1 else path)
        plt.close()
        pos += d.domain_size


def plot_z(all_data, z, title, path):
    if torch.is_tensor(z):  # when we do not use PCA, i.e. latent-size == 2
        dim_z = z.size()[1]
    else:  # when we use PCA, we return numpy
        dim_z = z.shape[1]

    assert dim_z <= 2, "can only plot z, if its dim is <=2"

    s = all_data[:, -1].unsqueeze(-1).int().numpy()
    data = np.concatenate((z, s), axis=1)

    if dim_z == 2:
        df = pd.DataFrame(data=data, columns=["z0", "z1", "s"])
    elif dim_z == 1:
        df = pd.DataFrame(data=data, columns=["z0", "s"])
    else:
        raise IndexError

    # df that only contains those rows where s=0
    df_s0 = df[df['s'] == -1]
    df_s1 = df[df['s'] == 1]
    # dont plot all data, otherwise to clotted
    # Return a random sample of items from an axis of object.
    df_s0 = df_s0.sample(min(1250, len(df_s0)))
    df_s1 = df_s1.sample(min(1250, len(df_s1)))
    # df has two columns, one with s=0, one s=1
    df = pd.concat([df_s0, df_s1])
    groups = df.groupby("s")
    for name, group in groups:
        if dim_z == 2:
            plt.scatter(group["z0"], group["z1"], label=name, s=20, alpha=0.3)
        else:
            if name == 0.0:
                plt.scatter(group["z0"], pd.DataFrame(np.zeros(group["z0"].shape)), label=name, s=20, alpha=0.3)
            else:
                plt.scatter(group["z0"], pd.DataFrame(np.ones(group["z0"].shape)), label=name, s=20, alpha=0.3)

    plt.suptitle(title)
    plt.legend()
    plt.savefig(path)
    plt.close()

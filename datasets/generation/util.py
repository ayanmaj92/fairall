
from numpy.random import RandomState
import numpy as np
import pandas as pd
import sklearn.metrics as skm

def get_random(seed=None):
    if seed is None:
        return RandomState()
    else:
        s = RandomState(seed)
        return s

def get_utility(d, y):
    d = pd.Series(d)
    y = pd.Series(y)
    c = 0.5
    u = d*(y - c)
    if len(u)==1:
        u = u[0]
    avu = np.average(u)
    return u, avu

def whiten(data, columns=None, conditioning=1e-8):
    """
    Whiten various datasets in data dictionary.
    Args:
        data: Data array.
        columns: The columns to whiten. If `None`, whiten all.
        conditioning: Added to the denominator to avoid divison by zero.
    """
    if columns is None:
        columns = np.arange(data.shape[1])
    mu = np.mean(data[:, columns], 0)
    std = np.std(data[:, columns], 0)
    data[:, columns] = (data[:, columns] - mu) / (std + conditioning)
    return data

def false_positive_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the false positive rate (also called fall-out)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return fpr

import numpy as np
import pandas as pd
from operator import add
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy.stats import norm
import math


def sigmoid(x, k=0, r=1):
    return 1 / (1 + np.exp(r * (-x - k)))

class BaseDistribution(object):

    def _sample_test_dataset(self, n_test, random):
        raise NotImplementedError("Subclass must override _sample_test_dataset(self, n_test).")

    def _sample_train_dataset(self, n_train, random):
        raise NotImplementedError("Subclass must override _sample_train_dataset(self, n_train).")

    # def sample_test_dataset(self, n_test, seed=None):
    def sample_test_dataset(self, n_test, seed):
        # return self._sample_test_dataset(n_test, get_random(seed) if seed else get_random())
        return self._sample_test_dataset(n_test, 13*seed)

    def sample_stats_dataset(self, n_test, seed):
        # return self._sample_test_dataset(n_test, get_random(seed) if seed else get_random())
        return self._sample_stats_dataset(n_test, 19*seed)

    # def sample_train_dataset(self, n_train, seed=None):
    def sample_train_dataset(self, n_train, seed):
        # return self._sample_train_dataset(n_train, get_random(seed) if seed else get_random())
        return self._sample_train_dataset(n_train, seed)


class GenerativeDistribution(BaseDistribution):
    def __init__(self, fraction_protected):


            # how many s=1 persons, i.e. 0.5 or 0.3
        self.fraction_protected = fraction_protected

        # standartized features
        # GPA
        self.muG = 0
        self.sigG = 0.8

        #LSAT
        self.muL = 0
        self.sigL = 1.

        #FYA, standardized
        self.muF = 0
        self.sigF = 0.3
        # to compute y
        self.threshold = 0

        # weight of mix in sigma, changed to 0
        self.r = 1 # not relevant for SCB


        # sensitive attribute
        self.wSF = -0.8 # changed from -0.8
        self.wSG = -1.5
        self.wSL = -0.5

        #Knowledge
        # self.muK = 1
        self.muK0 = -1
        self.muK1 = 1
        self.sigK = 0.6

        self.wKG = 1.5
        self.wKL = 1
        self.wKF = 1.3


    def _sample_features(self, n, fraction_protected, random):
        raise NotImplementedError("Subclass must override sample_features(self, n).")

    def _sample_labels(self, x, s, random):
        raise NotImplementedError("Subclass must override sample_labels(self, x, s).")

    def _sample_train_dataset(self, n_train, seed):
        xG, xL, s, k, k_mean, k_var = self._sample_features(n_train, self.fraction_protected, seed)
        # Todo: /// verify if its the same, if we used x in real dataset as DataFrame
        print(xG)
        xG_std = preprocessing.scale(xG)
        xL_std = preprocessing.scale(xG)
        y, y_fair, prob_FYA, fya = self._sample_labels(xG_std, xL_std, s, k, n_train, seed)

        s = s.astype('int32')
        y = y.astype('int32')
        d = {'GPA': xG, 'LSAT': xL, 'Y': y, 'S': s, 'Y_F': y_fair, 'prob_FYA': prob_FYA,
             'K': k, 'FYA': fya}
        # d = {'GPA': xG, 'LSAT': xL, 'Y': y, 'S': s}
        return pd.DataFrame(data=d)

    def _sample_test_dataset(self, n_test, seed):
        return self._sample_train_dataset(n_test, seed)

    # def _sample_stats_dataset(self, n_test, seed):
    #     xG, xL, s, k = self._sample_features(n_test, self.fraction_protected, seed)
    #     # Todo: /// verify if its the same, if we used x in real dataset as DataFrame
    #     xG_std = preprocessing.scale(xG)
    #     xL_std = preprocessing.scale(xG)
    #     y, y_fair, fya = self._sample_labels(xG_std, xL_std, s, k, n_test, seed)
    #
    #     s = s.astype('int32')
    #     y = y.astype('int32')
    #     d = {'S': s, 'GPA': xG, 'LSAT': xL, 'Y': y, 'Y_F':y_fair, 'K':k, 'FYA':fya}
    #     return pd.DataFrame(data=d)

class NoConfounders(GenerativeDistribution):

    def __init__(self, fraction_protected):
        super(NoConfounders, self).__init__(fraction_protected=fraction_protected)

    def _sample_features(self, n, fraction_protected, seed):
        random = np.random.RandomState(seed)
        random1 = np.random.RandomState(3*seed)
        random2 = np.random.RandomState(4*seed)

        number = random.rand(n, 1)
        s = (number < fraction_protected).astype(int)
        k = np.zeros(n).squeeze()
        epsG = random1.normal(0, self.sigG, n)
        epsL = random2.normal(0, self.sigL, n)
        xG = self.muG + epsG
        xL = self.muL + epsL
        return xG.squeeze(), xL.squeeze(), s.squeeze(), k

    def _sample_labels(self, xG, xL, s, k, n, seed):
        random = np.random.RandomState(6*seed)

        epsF = random.normal(0, self.sigF, n)
        xF = self.muF + epsF
        y = (xF > self.threshold).astype(int).squeeze()

        return y

class NonSensitiveConfounder(GenerativeDistribution):
    def __init__(self, fraction_protected):
        super(NonSensitiveConfounder, self).__init__(fraction_protected=fraction_protected)

    def _sample_features(self, n, fraction_protected, seed):
        random = np.random.RandomState(seed)
        random1 = np.random.RandomState(3*seed)
        random2 = np.random.RandomState(4*seed)
        random3 = np.random.RandomState(5*seed)

        # random sensitive label
        s = (random.rand(n, 1) < fraction_protected).astype(int).squeeze()
        # Knowledge
        k = random3.normal(self.muK, self.sigK, n)
        # epsilon
        epsG = random1.normal(0, self.sigG, n)
        epsL = random2.normal(0, self.sigL, n)
        # GPA
        xG = self.muG + (self.wKG * k).squeeze() + epsG
        # LSAT
        xL = self.muL + (self.wKG * k).squeeze()+ epsL

        return xG, xL, s, k

    def _sample_labels(self, xG, xL, s, k, n, seed):
        random = np.random.RandomState(6*seed)
        epsF = random.normal(0, self.sigF, n) # self.wK

        xF = self.muF + (self.wKF * k).squeeze() + epsF

        y = (xF > self.threshold).astype(int).squeeze()



        return y


class SensitiveConfounderBias(GenerativeDistribution):
    def __init__(self, fraction_protect, threshold=0.5, sigmoid_k=0, sigmoid_r=1):
        super().__init__(fraction_protect)
        self.threshold = threshold
        self.sigmoid_k = sigmoid_k
        self.sigmoid_r = sigmoid_r

    def _sample_features(self, n, fraction_protected, seed):
        random = np.random.RandomState(seed)
        random1 = np.random.RandomState(3*seed)
        random2 = np.random.RandomState(4*seed)
        random3 = np.random.RandomState(5* seed)
        random4 = np.random.RandomState(5 * seed)


        # random sensitive label
        s = (random.rand(n, 1) < fraction_protected).astype(int).squeeze()
        # Knowledge
        k_group = (random4.rand(n, 1) < 0.6).astype(int).squeeze()

        index_k1 = np.where(k_group == 1)

        muK = np.ones(len(s))*self.muK0
        muK[index_k1] = self.muK1

        k = random3.normal(muK, self.sigK, n)
        k_var = random3.normal(0, self.sigK, n)

        # epsilon
        # here the confounder only affects G, not L or F
        epsG = random1.normal(0, self.sigG, n)
        epsL = random2.normal(0, self.sigL, n)

        # # convert to -1
        s = np.where(s == 0, -1, s)
        # LSAT
        mLSAT_lin = self.muL + (self.wSL * s).squeeze() + (self.wKL * k).squeeze()
        xL = (1/(1 + np.exp(-mLSAT_lin))-0.5)*2 + epsL
        # GPA
        mGPA_lin = self.muG + (self.wSG * s).squeeze() + (self.wKG * k).squeeze()
        xG = (1/(1 + np.exp(-mGPA_lin))-0.5)*2 + epsG
        #
        # # convert back
        s = np.where(s == -1, 0, s)
        return xG, xL, s, k, muK, k_var

    def _sample_labels(self, xG, xL, s, k, n, seed):
        random = np.random.RandomState(6*seed)
        epsF = random.normal(0, self.sigF, n)

        # convert to -1
        s = np.where(s == 0, -1, s)

        xF = self.muF + (self.wSF * s).squeeze() + (self.wKF * k).squeeze() + epsF
        print('xF', xF)

        # loc_F = self.muF + np.add((self.wSF * s).squeeze(), (self.wKF * k).squeeze())
        # prob_FYA = norm.sf(self.threshold, loc=loc_F, scale=self.sigF)
        # y = (xF > self.threshold).astype(int).squeeze()
        # y_fair = (k > self.threshold).astype(int).squeeze()

        # Modifying prob_FYA and label computation to sigmoid.
        prob_FYA = sigmoid(xF, self.sigmoid_k, self.sigmoid_r)
        y = (prob_FYA > self.threshold).astype(int).squeeze()
        prob_fair_FYA = sigmoid(k, self.sigmoid_k, self.sigmoid_r)
        y_fair = (prob_fair_FYA > self.threshold).astype(int).squeeze()

        return y, y_fair, prob_FYA, xF


class SensitiveConfounderVar(GenerativeDistribution):
    def _sample_features(self, n, fraction_protected, seed):
        random = np.random.RandomState(seed)
        random1 = np.random.RandomState(3*seed)
        random2 = np.random.RandomState(4*seed)
        random3 = np.random.RandomState(5* seed)
        # random sensitive label
        s = (random.rand(n, 1) < fraction_protected).astype(int).squeeze()
        # Knowledge
        k = random3.normal(self.muK, self.sigK, n)
        # epsilon
        # here the confounder only affects G, not L or F
        epsG = random1.normal(0, self.sigG+self.r*s, n)
        epsL = random2.normal(0, self.sigL, n)
        # GPA
        xG = self.muG + np.add(np.add((0* s).squeeze(), (self.wKG * k).squeeze()), epsG)
        # LSAT
        xL = self.muL + np.add(np.add((0 * s).squeeze(), (self.wKL * k).squeeze()), epsL)
        return xG, xL, s, k

    def _sample_labels(self, xG, xL, s, k, n, seed):
        random = np.random.RandomState(6*seed)
        epsF = random.normal(0, self.sigF + self.r*s, n)
        xF = self.muF + np.add(np.add((0 * s).squeeze(), (self.wKF * k).squeeze()), epsF)
        y = (xF > self.threshold).astype(int).squeeze()
        return y

class SensitiveConfounderVarBias(GenerativeDistribution):
    def _sample_features(self, n, fraction_protected, seed):
        random = np.random.RandomState(seed)
        random1 = np.random.RandomState(3*seed)
        random2 = np.random.RandomState(4*seed)
        random3 = np.random.RandomState(5* seed)

        # random sensitive label
        s = (random.rand(n, 1) < fraction_protected).astype(int).squeeze()
        # Knowledge
        k = random3.normal(self.muK, self.sigK, n)
        # epsilon
        # here the confounder only affects G, not L or F
        epsG = random1.normal(0, self.sigG+self.r*s, n)
        epsL = random2.normal(0, self.sigL, n)
        # GPA
        xG = self.muG + np.add(np.add((self.wSG * s).squeeze(), (self.wKG * k).squeeze()), epsG)
        # LSAT
        xL = self.muL + np.add(np.add((self.wSL * s).squeeze(), (self.wKL * k).squeeze()), epsL)
        return xG, xL, s, k

    def _sample_labels(self, xG, xL, s, k, n, seed):
        random = np.random.RandomState(6*seed)
        epsF = random.normal(0, self.sigF + self.r*s, n)
        xF = self.muF + np.add(np.add((self.wSF * s).squeeze(), (self.wKF * k).squeeze()), epsF)
        y = (xF > self.threshold).astype(int).squeeze()
        return y

class UncalibratedScore(GenerativeDistribution):
    """An distribution modelling an uncalibrated score."""
    def __init__(self, fraction_protected):
        super(UncalibratedScore, self).__init__(fraction_protected=fraction_protected)
        self.bound = 0.8
        self.width = 30.0
        self.height = 3.0
        self.shift = 0.1

    @property
    def feature_dimension(self):
        return 1

    def _pdf(self, x):
        """Get the probability of repayment."""
        num = (
                np.tan(x)
                + np.tan(self.bound)
                + self.height
                * np.exp(-self.width * (x - self.bound - self.shift) ** 4)
        )
        den = 2 * np.tan(self.bound) + self.height
        return num / den

    def _sample_features(self, n, fraction_protected, seed):
        random = np.random.RandomState(seed)

        s = (
                random.rand(n, 1) < fraction_protected
        ).astype(int)

        shifts = s - 0.5
        x = truncnorm.rvs(
            -self.bound + shifts, self.bound + shifts, loc=-shifts
        ).reshape(-1, 1)
        return x, s

    def _sample_labels(self, x, s, n_train, seed):
        random = np.random.RandomState(6 * seed)
        yprob = self._pdf(x)
        return np.expand_dims(random.binomial(1, yprob), axis=1)

class DummyDistribution1D(BaseDistribution):
    """A simple generative model of the true distribution."""

    def __init__(self, config):
        """
        Initialize the true distribution.
        Args:
            config: The configuration dictionary.
        """
        super().__init__(config)
        self.type = "custom1d"
        self.theta = np.array(config["theta"])
        self.feature_dim = len(self.theta)
        if "split_support" not in config["custom_tweaks"]:
            self.threshold = self._threshold
        self.is_1d = True

    def sample_features(self, n, **kwargs):
        """
        Draw examples only for the features of the true distribution.
        Args:
            n: The number of examples to draw.
        Returns:
            x: np.ndarray with the features of dimension (n, k), where k is
                either 1 or 2 depending on whether a constant is added
        """
        if self.config["protected_fraction"] is not None:
            s = (
                np.random.rand(n, 1) < self.config["protected_fraction"]
            ).astype(int)
            x = 3.5 * np.random.randn(n, 1) + 3 * (0.5 - s)
        else:
            s = np.full(n, np.nan)
            x = 3.5 * np.random.randn(n, 1)

        if self.config["protected_as_feature"]:
            x = np.concatenate((x, s.reshape(-1, 1)), axis=1)
        if self.config["add_constant"]:
            x = np.hstack([np.ones([n, 1]), x])
        return x, s.ravel()

    def sample_labels(self, x, s, yproba=False):
        """
        Draw examples of labels for given features.
        Args:
            x: Given features (usually obtained by calling `sample_features`).
            s: Sensitive attribute.
            yproba: Whether to return the probabilities of the binary labels.
        Returns:
            y: np.ndarray of binary (0/1) labels (if `yproba=False`)
            y, yproba: np.ndarrays of binary (0/1) labels as well as the
                original probabilities of the labels (if `yproba=False`)
        """
        yprob = utils.sigmoid(x.dot(self.theta))

        if "bump_left" in self.config["custom_tweaks"]:
            yprob += np.exp(-(x[:, 1] + 6) ** 2 * 2) * 0.5
            yprob = np.maximum(np.minimum(yprob, 1), 0)
        if "bump_right" in self.config["custom_tweaks"]:
            yprob -= np.exp(-(x[:, 1] - 5) ** 2 * 0.8) * 0.35
            yprob = np.maximum(np.minimum(yprob, 1), 0)
        if "split_support" in self.config["custom_tweaks"]:
            yprob = 0.8 * utils.sigmoid(0.6 * (x[:, 1] + 3)) * utils.sigmoid(
                -5 * (x[:, 1] - 3)
            ) + utils.sigmoid(x[:, 1] - 5)

        y = np.random.binomial(1, yprob)
        if yproba:
            return y, yprob
        return y

    def _threshold(self, cost):
        """The threshold for this policy."""
        if len(self.theta) == 1:
            return 0.0
        if len(self.theta) == 2:
            return utils.get_threshold(self.theta, cost)
        else:
            raise RuntimeError("Scalar threshold exists only for 1D.")

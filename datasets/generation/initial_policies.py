
import numpy as np
import pandas as pd
import math
import sklearn.metrics as skm
from util import get_random, get_utility, false_positive_rate
from sklearn.linear_model import LogisticRegression

class LogPolicy(object):
    def __init__(self, poltype, seed):
        self.classifier = LogisticRegression(solver='liblinear', fit_intercept=True)
        self.fitted = False
        self.poltype = poltype  # can be "Unaware" or "Full"
        self.seed = seed

    def fit(self, data):
        if self.poltype == "Unaware":
            data = data.drop(['S'], axis = 1)
        y = data['Y']
        X = data.drop(['Y'], axis = 1)

        self.classifier.fit(X, y)
        self.fitted = True

    def predict(self, data):
        assert self.fitted == True, "Fit policy before predicting"

        data1 = data
        if self.poltype == "Unaware":
            data1 = data1.drop(['S'], axis = 1)

        y = data1['Y']
        X = data1.drop(['Y'], axis = 1)

        prob = self.classifier.predict_proba(X)[:,1]

        _, average_util = get_utility(prob, y)

        fprs = []
        for i in range(10):
            random_s = np.random.RandomState(self.seed*i)
            dec = random_s.binomial(1, prob, len(prob))
            #todo: this is outcommented and fpr set to 0, because it didnt work
            # fpr = false_positive_rate(y, dec)
            fpr = 0
            fprs.append(fpr)

        return prob, average_util, np.mean(fprs)


class OptimalPolicy(object):
    def __init__(self, prob, seed):
        self.prob = prob
        self.seed = seed

    def predict(self, data):
        y = data['Y']

        index_y1 = np.where(y == 1)
        index_y0 = np.where(y == 0)

        prob = pd.Series(np.zeros(len(y)), name='OPT')
        prob.loc[index_y1] = self.prob
        prob.loc[index_y0] = 1-self.prob

        _, average_util = get_utility(prob, y)

        fprs = []
        for i in range(10):
            random_s = np.random.RandomState(self.seed*i)
            dec = random_s.binomial(1, prob, len(prob))
            fpr = false_positive_rate(y, dec)
            fprs.append(fpr)

        return prob, average_util, np.mean(fprs)


class BiasedPolicy(object):
    def __init__(self, prob, seed):
        self.prob = prob
        self.bias_s = 0.15
        self.prob_s0 = min(0.99,self.prob+self.bias_s)
        self.prob_s1 = max(0.01, self.prob - self.bias_s)
        self.seed = seed

    def predict(self, data):
        y = data['Y']
        s = data['S']

        index_y0 = y[y == 0].index.values
        index_y1s1 = data.index[(data['Y'] == 1) & (data['S'] ==-1)].tolist()
        index_y1s0 = data.index[(data['Y'] == 1) & (data['S'] ==1)].tolist()

        prob = pd.Series(np.ones(len(y)), name='BP')
        prob.loc[(index_y1s1)] = self.prob_s1
        prob.loc[(index_y1s0)] = self.prob_s0
        prob.loc[(index_y0)] = 1 - self.prob

        _, average_util = get_utility(prob, y)

        fprs = []
        for i in range(10):
            random_s = np.random.RandomState(self.seed*i)
            dec = random_s.binomial(1, prob, len(prob))
            fpr = false_positive_rate(y, dec)
            fprs.append(fpr)

        return prob, average_util, np.mean(fprs)


class SigmoidPolicy(object):
    def __init__(self, pol_type, dist, seed, biased=False):
        self.dist = dist
        self.biased = biased
        self.pol_type = pol_type
        self.seed = seed

    def predict(self, data):
        y = data['Y']

        def sigmoid(x, w0, w1):
            return 1 / (1 + math.exp(-(w1 * x + w0)))

        def pol_sigmoid(x, pol_type='lenient'):
            if pol_type == 'lenient':
                w0, w1 = 1, 1
            elif pol_type == 'harsh':
                w0, w1 = -3, 2
            return sigmoid(x, w0, w1)

        def bpol_sigmoid(x, s, pol_type):
            if pol_type == 'lenient':
                w0, w1 = 1, 1
            elif pol_type == 'harsh':
                w0, w1 = -3, 2
            if s == 1:
                w0 += -1.5  # stricker policy
            elif s == -1:
                w0 += 1.5
            return sigmoid(x, w0, w1)

        if self.dist in ['SCB', 'NSC']: #todo: add other synthetic datasets
           x = data['GPA']
           if self.biased==False:
              prob = [pol_sigmoid(i, self.pol_type) for i in x]
           else:
               s = data['S']
               prob = [bpol_sigmoid(i[0], i[1], str(self.pol_type)) for i in zip(x, s)]
        else:
            print ('dist is not implemented:', self.dist)

        _, average_util = get_utility(prob, y)

        fprs = []
        for i in range(10):
            random_s = np.random.RandomState(self.seed*i)
            dec = random_s.binomial(1, prob, len(prob))
            fpr = false_positive_rate(y, dec)
            fprs.append(fpr)

        return prob, average_util, np.mean(fprs)

class RandomPolicy(object):
    def __init__(self, seed):
        self.seed = seed

    def predict(self, data):
        y = data['Y']

        prob = pd.Series(np.ones(len(y))*0.5, name='RAN')

        _, average_util = get_utility(prob, y)

        fprs = []
        for i in range(10):
            random_s = np.random.RandomState(self.seed*i)
            dec = random_s.binomial(1, prob, len(prob))
            fpr = false_positive_rate(y, dec)
            fprs.append(fpr)

        return prob, average_util, np.mean(fprs)
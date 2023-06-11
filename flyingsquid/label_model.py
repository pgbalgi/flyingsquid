from flyingsquid.helpers import *
from flyingsquid import _triplets
from flyingsquid import _graphs
from flyingsquid import _observables
from flyingsquid import _lm_parameters
import numpy as np
import itertools

class LabelModel(_triplets.Mixin, _graphs.Mixin, _observables.Mixin,
                 _lm_parameters.Mixin):
    
    def __init__(self, class_balance):
        self.acc = None
        self.abstain_rate = None
        self.class_balance = class_balance
    
    # Make this Picklable
    def save(obj):
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj
        
    def fit(self, L_train):
        '''Compute the marginal probabilities of each clique and separator set in the junction tree.
        
            L_train: an n x m x d matrix of LF outputs. L_train[k][i] is the value of \lambda_i on item k.
                1 means positive, -1 means negative, 0 means abstain.
            class_balance: a vector of the probability of each class.
            verbose: if True, print out messages to stderr as we make progress

            Outputs: None.
        '''
        n, m, d = L_train.shape

        self.abstain_rate = 1 - np.mean(np.abs(np.prod(L_train, axis=-1)), axis=0)
        self.acc = np.zeros(m)


        moments = np.full((m,m), np.nan)
        for i in range(m):
            for j in range(i+1,m):
                moments[i,j] = self._compute_moment(L_train[:,i], L_train[:,j])
                moments[j,i] = moments[i,j]
        

        for i in range(m):
            acc_i = []

            for k, l in itertools.combinations(set(range(m)) - {i}, 2):
                acc_kl = np.sqrt(np.abs(moments[i,k] * moments[i,l] / moments[k,l]))
                if not np.isnan(acc_kl):
                    acc_i.append(acc_kl)

            self.acc[i] = np.mean(acc_i)



    def _compute_moment(self, lambda_i, lambda_j):
        non_abstain = np.nonzero(np.prod(lambda_i * lambda_j, axis=-1))
        if non_abstain[0].size == 0:
            return np.nan

        lambda_i = lambda_i[non_abstain]
        lambda_j = lambda_j[non_abstain]

        return self.class_balance @ np.mean(lambda_i * lambda_j, axis=0)

    
    def predict_proba(self, L_matrix):
        '''Predict the probabilities of the Y's given the outputs of the LF's.
        
        L_matrix: a n x m x d matrix of of LF outputs. L_matrix[k][i] is the value of \lambda_i on item k.
            1 means positive, -1 means negative, 0 means abstain.
                
        Outputs: a d vector of probabilities
        '''
        n, m, d = L_matrix.shape
        proba = np.zeros((n,d))

        abstain = np.nonzero(np.prod(L_matrix, axis=-1) == 0)

        for c in range(d):
            y = np.zeros(d)
            y[c] = 1

            match = L_matrix @ y
            # mismatches = np.nonzero(match == -1)

            prob_c = (1 + match * self.acc) / 2
            # prob_c[mismatches] *= self.class_balance / (1 - self.class_balance[c])
            prob_c *= 1 - self.abstain_rate
            prob_c[abstain] = self.abstain_rate[abstain[-1]]

            proba[:,c] = np.prod(prob_c, axis=-1) * self.class_balance[c]


        proba /= np.sum(proba, axis=-1, keepdims=True)

        return proba
    

    def predict(self, L_matrix):
        '''Predict the value of the Y's that best fits the outputs of the LF's.
        
        L_matrix: a n x m x d matrix of LF outputs. L_matrix[k][i] is the value of \lambda_i on item k.
            1 means positive, -1 means negative, 0 means abstain.
                
        Outputs: a d matrix of predicted outputs.
        '''
        n, m, d = L_matrix.shape
        one_hot_pred = np.zeros((n,d))

        pred = np.argmax(self.predict_proba(L_matrix), axis=-1)
        one_hot_pred[range(n), pred] = 1

        return one_hot_pred
    


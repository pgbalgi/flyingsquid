from flyingsquid.helpers import *
from flyingsquid import _triplets
from flyingsquid import _graphs
from flyingsquid import _observables
from flyingsquid import _lm_parameters
import numpy as np

class LabelModel(_triplets.Mixin, _graphs.Mixin, _observables.Mixin,
                 _lm_parameters.Mixin):
    
    def __init__(self, class_balance):
        self.acc = None
        # self.lambda_params = None
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
        self.acc = np.zeros(m)
        # self.lambda_params = np.zeros((m,d))
        self.abstain_rate = np.zeros(m)

        for i in range(m):
            acc_i = []
            lambda_i = L_train[:,i]

            abstain = np.unique(np.where(lambda_i[:,] == 0)[0])
            self.abstain_rate[i] = abstain.size / n

            ks = set(range(m))
            ks.remove(i)
            for k in ks:
                lambda_k = L_train[:,k]
                moment_ik = self._compute_moment(lambda_i, lambda_k)

                ls = set(ks)
                ls.remove(k)
                for l in ls:
                    lambda_l = L_train[:,l]
                    moment_il = self._compute_moment(lambda_i, lambda_l)
                    moment_kl = self._compute_moment(lambda_k, lambda_l)

                    acc_kl = np.sqrt(np.abs(moment_ik * moment_il / moment_kl))
                    acc_i.append(acc_kl)
            

            self.acc[i] = np.mean(acc_i)
     


    def _compute_moment(self, lambda_i, lambda_j):
        non_abstain_i = np.unique(np.where(lambda_i[:,] != 0)[0])
        non_abstain_j = np.unique(np.where(lambda_j[:,] != 0)[0])
        non_abstain = np.intersect1d(non_abstain_i, non_abstain_j)

        lambda_i = lambda_i[non_abstain]
        lambda_j = lambda_j[non_abstain]

        return self.class_balance @ np.mean(lambda_i * lambda_j, axis=0)

    
    def predict_proba(self, L_matrix):
        '''Predict the probabilities of the Y's given the outputs of the LF's.
        
        L_matrix: a m x d matrix of of LF outputs. L_matrix[k][i] is the value of \lambda_i on item k.
            1 means positive, -1 means negative, 0 means abstain.
                
        Outputs: a d vector of probabilities
        '''
        m, d = L_matrix.shape
        proba = np.zeros(d)

        abstain = np.unique(np.where(L_matrix[:,] == 0)[0])

        for c in range(d):
            y = np.zeros(d)
            y[c] = 1

            prob_c = (1 + (L_matrix @ y) * self.acc) / 2
            prob_c *= 1 - self.abstain_rate
            prob_c[abstain] = self.abstain_rate[abstain]

            proba[c] = np.prod(prob_c) * self.class_balance[c]

        proba /= np.sum(proba)

        return proba
    

    def predict(self, L_matrix):
        '''Predict the value of the Y's that best fits the outputs of the LF's.
        
        L_matrix: a n x m x d matrix of LF outputs. L_matrix[k][i] is the value of \lambda_i on item k.
            1 means positive, -1 means negative, 0 means abstain.
                
        Outputs: a d matrix of predicted outputs.
        '''
        n, m, d = L_matrix.shape
        one_hot_pred = np.zeros((n,d))

        for i in range(n):
            pred = np.argmax(self.predict_proba(L_matrix[i]), axis=-1)
            one_hot_pred[i,pred] = 1

        return one_hot_pred
    


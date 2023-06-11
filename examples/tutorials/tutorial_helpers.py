import numpy as np
from numpy.random import seed, rand
import itertools

def exponential_family (lam, y, theta, theta_y):
    # without normalization
    return np.exp(theta_y @ y + theta @ lam @ y)

# create vector describing cumulative distribution of lambda_1, ... lambda_m, Y
def make_pdf(d, theta, theta_y, lst):
    p = np.zeros(len(lst))
    for i in range(len(lst)):
        labels = np.array(lst[i])
        lam = make_lambda(labels[:-1], d)
        y = make_y(labels[-1], d)

        p[i] = exponential_family(lam, y, theta, theta_y)
        
    return p/sum(p)

def make_cdf(pdf):
    return np.cumsum(pdf)

def make_lambda(labels, d):
    lam = np.full((labels.size, d), -1, dtype=int)
    lam[range(labels.size), labels.flatten()] = 1
    return lam.reshape(labels.shape + (d,))

def make_y(labels, d):
    y = np.zeros((labels.size, d), dtype=int)
    y[range(labels.size), labels.flatten()] = 1
    return y.reshape(labels.shape + (d,))

# draw a set of lambda_1, ... lambda_m, Y based on the distribution
def sample(lst, cdf):
    r = np.random.random_sample()
    smaller = np.where(cdf < r)[0]
    if len(smaller) == 0:
        i = 0
    else:
        i = smaller.max() + 1
    return lst[i]

def generate_data(n, theta, m, d, theta_y=None):
    v = m+1
    if theta_y is None:
        theta_y = np.full(d, 1/d)
    
    lst = list(map(list, itertools.product(range(d), repeat=v)))
    pdf = make_pdf(d, theta, theta_y, lst)
    cdf = make_cdf(pdf)

    sample_matrix = np.zeros((n,v), dtype=int)
    for i in range(n):
        sample_matrix[i,:] = sample(lst,cdf)
        
    return sample_matrix

def synthetic_data_basics(d=2):
    seed(0)
    
    n_train = 10000
    n_dev = 500
    
    m = 5
    theta = [1.5,.5,.2,.2,.05]
    abstain_rate = [.8, .88, .28, .38, .45]
    
    train_data = generate_data(n_train, theta, m, d)
    dev_data = generate_data(n_dev, theta, m, d)
    
    L_train = make_lambda(train_data[:,:-1], d)
    L_dev = make_lambda(dev_data[:,:-1], d)
    Y_dev = make_y(dev_data[:,-1], d)

    train_values = rand(n_train * m).reshape(L_train.shape[:-1])
    dev_values = rand(n_dev * m).reshape(L_dev.shape[:-1])
    
    L_train[train_values < (abstain_rate,) * n_train, :] = 0
    L_dev[dev_values < (abstain_rate,) * n_dev, :] = 0
    
    return L_train, L_dev, Y_dev

def print_statistics(L_dev, Y_dev):
    m = L_dev.shape[1]
    
    for i in range(m):
        acc = np.sum(L_dev[:,i] == Y_dev)/np.sum(L_dev[:,i] != 0)
        abstains = np.sum(L_dev[:,i] == 0)/Y_dev.shape[0]
        
        print('LF {}: Accuracy {}%, Abstain rate {}%'.format(
            i, int(acc * 100), int((abstains) * 100)))
import numpy as np
import torch
import torch.nn as nn
from notears.nonlinear import notears_nonlinear
from notears.nonlinear import NotearsMLP
import utils as ut
import time as t


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # load data
    X = np.loadtxt('X.csv', delimiter=',')
    W_true = np.loadtxt('W_true.csv', delimiter=',')
    n, d = X.shape

    # notears nonlinear
    start_time = t.time()
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=0, lambda2=0) # no l1 and l2 constraint version
    end_time = t.time()
    assert ut.is_dag(W_est)
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    SHD, extra, missing, reverse = ut.count_accuracy(W_true, W_est != 0)
    print(f'SHD: {SHD}, extra: {extra}, missing: {missing}, reverse: {reverse}, time: {end_time - start_time: .4f}.')


def main_syn():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # generate synthetic data
    ut.set_random_seed(123)
    n, d, s0, graph_type, sem_type = 1000, 5, 9, 'ER', 'mlp'
    W_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt('W_true.csv', W_true, delimiter=',')
    X = ut.simulate_nonlinear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')


if __name__ == '__main__':
    # main_syn()
    main()

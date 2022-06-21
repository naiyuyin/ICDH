import numpy as np
# import mynumpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import utils as ut
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import igraph as ig
import torch
from torch import nn
import time


def nll(target, mu, var):
    """
    Probabilistic Loss: negative log likelihood loss
    target: input n*d
    mu: mean of conditional probability distribution, n*d
    var: variance of conditional probability distribution, n*d
    """
    R = target - mu
    nll_loss = 0.5 * torch.sum(torch.log(2 * np.pi * var) + R ** 2 / var)
    return nll_loss

def rec(target, mu):
    n = target.shape[0]
    R = target - mu
    rec_loss = 0.5 / n * torch.sum(R ** 2)
    return rec_loss

def _h(W):
    """
    The continuous DAG constraint (Zheng et al, 2018)
    W: weighted adjacency matrix, d*d
    """
    d = W.shape[0]
    M = np.eye(d) + W * W / d
    E = np.linalg.matrix_power(M, d - 1)
    h = (E.T * M).sum() - d
    G_h = E.T * W * 2
    return h, G_h


def nll_linear_A(X, lamb=0, verbose=False):
    """
    test algorithm:
    use one set of parameter A for both mean and variance.
    X: input, n*d
    lamb: coefficient of uncertainty regularization
    """
    def _func(params):
        # convert parameters to A, B, B0
        A = (params[:d * d] - params[d * d:2 * d * d]).reshape([d, d])
        A0 = params[2 * d * d:].reshape([1, d])

        # compute NLL loss and pose acyclicity constraint on A
        var = np.exp(X_n @ A + A0)
        R = X - X @ A
        loss = 0.5 * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_A, G_h_A = _h(A)
        # objective = NLL loss + acyclicity penalty for A(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho * h_A * h_A + alpha * h_A + lamb * np.sum(var)

        # Compute graident of A,B,B0, since only updates A, gradients of B and B0 should be 0.
        G_loss_A = 0.5 * (X_n.T @ (1 - R ** 2 / var)) - X.T @ (R / var) + lamb * X_n.T @ var
        G_loss_A0 = 0.5 * (1 - R ** 2 / var).sum(axis=0) + lamb * np.sum(var, axis=0)

        # back-propagate DAG constraint on gradient of A
        G_loss_A = G_loss_A + (rho * h_A + alpha) * G_h_A
        g_obj = np.concatenate((G_loss_A, -G_loss_A, G_loss_A0), axis=None)
        return obj, g_obj

    start_time = time.time()
    n, d = X.shape
    c = 0.25
    s = 10
    h_tol = 1e-8
    rho_max = 1e+16

    # centering
    X_n = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True) #normalize input X for variance estimation
    X = X - np.mean(X, axis=0, keepdims=True) # centering X remove the effect of bias A0

    # Initialize parameters and bounds
    A_est = np.random.normal(0, 0.0001, size=(2 * d * d))
    A0_est = np.random.normal(0, 1e-16, size=(1,d))
    rho, alpha, h_A = 1.0, 0.0, np.inf
    params_est = np.concatenate((A_est, A0_est), axis=None)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)] + [(None, None) for i in range(d)]
    losses = []
    losses.append(0.5 * (np.log(2 * np.pi * np.exp(X_n @ (A_est[:d*d] - A_est[d*d:]).reshape(d,d) + A0_est)) + (X - X @ (A_est[:d*d] - A_est[d*d:]).reshape(d,d)) ** 2 / np.exp(X_n @ (A_est[:d*d] - A_est[d*d:]).reshape(d,d) + A0_est)).sum())
    
    # Augmented Lagrangian Method
    while h_A > h_tol and rho < rho_max:
        while rho < rho_max:
            sol = sopt.minimize(_func, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 1})
            params_new = sol.x
            A_new, A0_new = (params_new[:d * d] - params_new[d * d: 2 * d * d]).reshape([d, d]), params_new[2 * d * d:].reshape([1, d])
            h_new_A, _ = _h(A_new)
            if h_new_A < c * h_A:
                break
            else:
                rho = rho * s
        params_est, h_A = params_new, h_new_A
        alpha += rho * h_A
    
    A_est, A0_est = (params_est[:d * d] - params_est[d * d: 2 * d * d]).reshape([d, d]), params_est[2 * d * d:].reshape([1, d])
    nll = 0.5 * (np.log(2 * np.pi * np.exp(X_n @ A_est + A0_est)) + (X - X @ A_est) ** 2 / np.exp(X_n @ A_est + A0_est)).sum()
    rec = 0.5 / n * ((X - X @ A_est) ** 2).sum()
    end_time = time.time()
    var_est = np.exp(X_n @ A_est + A0_est)
    return A_est, A0_est, var_est, nll, rec, end_time - start_time
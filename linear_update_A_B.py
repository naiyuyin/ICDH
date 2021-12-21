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

def update_A(X, X_n, params_est, bnds, c = 0.25, s = 10, lamb = 0, h_tol = 1e-8, rho_max = 1e+16):
    """
    Subrountine updateA: fix values of B, B0 and only update A in NLL loss subject to the acyclicity constraint using quasi-newton optimizer: lbfgsb
    X: input, n*d
    X_n: normalized input for variances estimation, n*d
    params_est: A(2*d*d), B(d*d), B0(1*d)
    bnds: boundary
    lamb: the coefficient for uncertainty regularization
    """
    def _func_A(params):
        # convert parameters to A, B, B0
        A = (params[:d * d] - params[d * d:2 * d * d]).reshape([d, d])
        B = params[2 * d * d:3 * d * d].reshape([d, d])
        B0 = params[3 * d * d:].reshape([1, d])

        # compute NLL loss and pose acyclicity constraint on A
        var = np.exp(X_n @ B + B0)
        R = X - X @ A
        loss = 0.5 * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_A, G_h_A = _h(A)
        # objective = NLL loss + acyclicity penalty for A(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho_A * h_A * h_A + alpha_A * h_A + lamb * np.sum(var)

        # Compute graident of A,B,B0, since only updates A, gradients of B and B0 should be 0.
        G_loss_A = - X.T @ (R / var)
        # G_loss_B =  0.5 * (X.T @ (1 - R ** 2 / var))
        # G_loss_B0 = 0.5 * (1 - R ** 2 / var).sum(axis=0)
        G_loss_B = np.zeros([d, d])
        G_loss_B0 = np.zeros([d, ])

        # back-propagate DAG constraint on gradient of A
        G_loss_A = G_loss_A + (rho_A * h_A + alpha_A) * G_h_A
        g_obj = np.concatenate((G_loss_A, -G_loss_A, G_loss_B, G_loss_B0), axis=None)
        # print("%f,%f,%f,%d,%d"%(loss, rec, h_A, rho, alpha))
        return obj, g_obj

    n,d = X.shape
    alpha_A, rho_A, h_A = 0, 1, np.inf
    while h_A > h_tol and rho_A < rho_max:
    # update A with fixed B, B0
        while rho_A < rho_max:
            sol = sopt.minimize(_func_A, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            A_new, B_new, B0_new = (params_new[:d * d] - params_new[d * d: 2 * d * d]).reshape([d, d]), params_new[2 * d * d:3 * d * d].reshape([d, d]), params_new[3 * d * d:].reshape([1, d])
            h_new_A, _ = _h(A_new)
            if h_new_A < c * h_A:
                break
            else:
                rho_A = rho_A * s
        params_est, h_A = params_new, h_new_A
        alpha_A += rho_A * h_A
    return params_est

def update_B(X, X_n, params_est, bnds, c = 0.25, s = 10, lamb = 0, h_tol = 1e-8, rho_max = 1e+16):
    """
        Subrountine updateA: fix values of A and only update B, B0 in NLL loss subject to the acyclicity constraint using quasi-newton optimizer: lbfgsb
        X: input, n*d
        X_n: normalized input for variances estimation, n*d
        params_est: A(2*d*d), B(d*d), B0(1*d)
        bnds: boundary
        lamb: the coefficient for uncertainty regularization
        """
    def _func_B(params):
        # convert parameters to A, B, B0
        A = (params[:d * d] - params[d * d:2 * d * d]).reshape([d, d])
        B = params[2 * d * d:3 * d * d].reshape([d, d])
        B0 = params[3 * d * d:].reshape([1, d])

        # compute NLL loss and pose acyclicity constraint on B
        var = np.exp(X_n @ B + B0)
        R = X - X @ A
        loss = 0.5 * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_B, G_h_B = _h(B)
        # objective = NLL loss + acyclicity penalty for B(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho_B * h_B * h_B + alpha_B * h_B + lamb * np.sum(var)

        # Compute graident of A,B,B0, since A is fixed, the gradient for A should be 0
        G_loss_A = np.zeros([d, d])
        G_loss_B = 0.5 * (X_n.T @ (1 - R ** 2 / var)) + lamb * X_n.T @ var
        G_loss_B0 = 0.5 * (1 - R ** 2 / var).sum(axis=0) + lamb * np.sum(var, axis = 0)

        # back-propagate DAG constraint on gradient of A
        G_loss_B = G_loss_B + (rho_B * h_B + alpha_B) * G_h_B
        g_obj = np.concatenate((G_loss_A, -G_loss_A, G_loss_B, G_loss_B0), axis=None)
        # print("%f,%f,%f,%d,%d"%(loss, rec, h_A, rho, alpha))
        return obj, g_obj

    n, d = X.shape
    alpha_B, rho_B, h_B = 0, 1, np.inf
    while h_B > h_tol and rho_B < rho_max:
        # update B, B0 with fixed A
        while rho_B < rho_max:
            sol = sopt.minimize(_func_B, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            A_new, B_new, B0_new = (params_new[:d * d] - params_new[d * d: 2 * d * d]).reshape([d, d]), params_new[2 * d * d:3 * d * d].reshape([d, d]), params_new[3 * d * d:].reshape([1, d])
            h_new_B, _ = _h(B_new)
            if h_new_B < c * h_B:
                break
            else:
                rho_B = rho_B * s
        params_est, h_B = params_new, h_new_B
        alpha_B += rho_B * h_B
    return params_est


def nll_linear_A_B(X, lamb = 0, verbose = False):
    """
    a two-phrase iterative DAG learning approach
    X: input, n*d
    lamb: coefficient of uncertainty regularization
    """
    start_time = time.time()
    n, d = X.shape
    c = 0.25
    s = 10
    iter = 1

    # centering
    X_n = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True) #normalize input X for variance estimation
    X = X - np.mean(X, axis=0, keepdims=True) # centering X remove the effect of bias A0

    # Initialize parameters and bounds
    A_est = np.random.normal(0, 0.0001, size=(2 * d * d))
    # B_est = np.zeros([d,d])
    # B0_est = np.zeros([1,d])
    B_est = np.random.normal(0,1e-16, size=(d,d))
    B0_est = np.random.normal(0, 1e-16, size=(1,d))
    # rho_A, alpha_A, rho_B, alpha_B, h_A, h_B = 1.0, 0.0, 1.0, 0.0, np.inf, np.inf
    params_est = np.concatenate((A_est, B_est, B0_est), axis=None)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)] +  [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)] + [(None, None) for i in range(d)]
    losses = []
    losses.append(0.5 * (np.log(2 * np.pi * np.exp(X_n @ B_est + B0_est)) + (X - X @ (A_est[:d*d] - A_est[d*d:]).reshape(d,d)) ** 2 / np.exp(X_n @ B_est + B0_est)).sum())
    # Augmented Lagrangian Method
    while(True):
        if verbose == True:
            print("Iteration %d:"%(iter))
            print("update A")
        params_new = update_A(X = X, X_n = X_n, params_est = params_est, bnds = bnds, c = c, s = s, lamb = lamb)
        if verbose == True:
            print("update B, B0")
        params_new = update_B(X = X, X_n = X_n, params_est = params_new, bnds = bnds, c = c, s = s, lamb = lamb)
        A_new, B_new, B0_new = (params_new[:d * d] - params_new[d * d: 2 * d * d]).reshape([d, d]), params_new[2 * d * d:3 * d * d].reshape([d, d]), params_new[3 * d * d:].reshape([1, d])
        losses.append(0.5 * (np.log(2 * np.pi * np.exp(X_n @ B_new + B0_new)) + ( X - X @ A_new)** 2 / np.exp(X_n @ B_new + B0_new)).sum())
        if verbose == True:
            print("Iteration %d loss: %f, previous loss: %f, difference of the losses: %f."%(iter, losses[-1], losses[-2], losses[-2] - losses[-1]))
        iter += 1
        if losses[-2] - losses[-1] < 10:
            # print(losses)
            break
        else:
            params_est, A_est, B_est, B0_est = params_new, A_new, B_new, B0_new
    end_time = time.time()
    var_est = np.exp(X_n @ B_est + B0_est)
    return A_est, B_est, B0_est, var_est, losses, end_time - start_time
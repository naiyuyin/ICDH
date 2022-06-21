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


def nll(X, X_n, A, a):
    """
    Probabilistic Loss: negative log likelihood loss
    target: input n*d
    mu: mean of conditional probability distribution, n*d
    var: variance of conditional probability distribution, n*d
    """
    var = np.exp(a * (X_n @ A))
    R = X - X @ A
    nll_loss = 0.5 * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
    return nll_loss


def rec(X, A):
    n = X.shape[0]
    R = X - X @ A
    rec_loss = 0.5 / n * np.sum(R ** 2)
    return rec_loss


def _h(W):
    """
    The continuous DAG constraint (Zheng et al, 2018)
    W: weighted adjacency matrix, d*d
    """
    d = W.shape[0]
    # M = np.eye(d) + W * W / d
    # E = np.linalg.matrix_power(M, d - 1)
    # h = (E.T * M).sum() - d
    E = slin.expm(W * W)
    h = np.trace(E) - d
    G_h = E.T * W * 2
    return h, G_h


def params2AB0(params, N):
    A = (params[: N * N] - params[N * N: 2 * N * N]).reshape([N, N])
    a = params[2 * N * N]
    B0 = params[2 * N * N + 1:].reshape([1, N])
    return A, a, B0


def update_A(X, X_n, params_est, bnds, c=0.25, s=10, lamb=0, h_tol=1e-8, rho_max=1e+16):
    """
    Subrountine updateA: fix values of B, B0 and only update A in NLL loss subject to the acyclicity constraint using quasi-newton optimizer: lbfgsb
    X: input, n*d
    X_n: normalized input for variances estimation, n*d
    params_est: A(2*d*d), a, B0(1*d)
    bnds: boundary
    lamb: the coefficient for uncertainty regularization
    """

    def _func_A(params):
        # convert parameters to A, B, B0
        A = (params[:d * d] - params[d * d:2 * d * d]).reshape([d, d])
        a = params[2 * d * d]
        # B0 = params[2 * d * d + 1:].reshape([1, d])

        # compute NLL loss and pose acyclicity constraint on A
        # var = np.exp(a * (X_n @ A) + B0)
        # var = np.exp(a * (X_n @ A + B0))
        var = np.exp(a * (X_n @ A))
        R = X - X @ A
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_A, G_h_A = _h(A)
        # objective = NLL loss + acyclicity penalty for A(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho_A * h_A * h_A + alpha_A * h_A + lamb * np.sum(var) / n

        # Compute graident of A,B,B0, since only updates A, gradients of B and B0 should be 0.
        G_loss_A = - X.T @ (R / var) / n + 0.5 * a * (X_n.T @ (1 - R ** 2 / var)) / n + lamb * a * X_n.T @ var / n
        G_loss_a = 0
        # G_loss_B0 = np.zeros([d, ])

        # back-propagate DAG constraint on gradient of A
        G_loss_A = G_loss_A + (rho_A * h_A + alpha_A) * G_h_A
        g_obj = np.concatenate((G_loss_A, -G_loss_A, G_loss_a), axis=None)
        return obj, g_obj

    n, d = X.shape
    alpha_A, rho_A, h_A = 0, 1, np.inf
    while h_A > h_tol and rho_A < rho_max:
        while rho_A < rho_max:
            sol = sopt.minimize(_func_A, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            A_new, a_new = (params_new[:d * d] - params_new[d * d: 2 * d * d]).reshape([d, d]), params_new[
                2 * d * d]#, params_new[2 * d * d + 1:].reshape([1, d])
            h_new_A, _ = _h(A_new)
            if h_new_A < c * h_A:
                break
            else:
                rho_A = rho_A * s
        params_est, h_A = params_new, h_new_A
        alpha_A += rho_A * h_A
    return params_est


def update_AaB0(X, X_n, params_est, bnds, fix_a=True, c=0.25, s=10, lamb=0, h_tol=1e-8, rho_max=1e+16):
    """
         X: input, n*d
        X_n: normalized input for variances estimation, n*d
        params_est: A(2*d*d), a, B0(1*d)
        bnds: boundary
        lamb: the coefficient for uncertainty regularization
        """

    def _func_AaB0(params):
        # convert parameters to A, a, B0
        A = (params[:d * d] - params[d * d:2 * d * d]).reshape([d, d])
        a = params[2 * d * d]
        # B0 = params[2 * d * d + 1:].reshape([1, d])

        # compute NLL loss and pose acyclicity constraint on A
        # var = np.exp(a * X_n @ A + B0)
        # var = np.exp(a * (X_n @ A + B0))
        var = np.exp(a * (X_n @ A))
        R = X - X @ A
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_B, G_h_B = _h(A)
        # objective = NLL loss + acyclicity penalty for B(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho_B * h_B * h_B + alpha_B * h_B + lamb * np.sum(var) / n

        # Compute graident of A,B,B0, since A is fixed, the gradient for A should be 0
        G_loss_A = - X.T @ (R / var)  / n + 0.5 * a * (X_n.T @ (1 - R ** 2 / var)) / n + lamb * a * X_n.T @ var / n
        if fix_a:
            G_loss_a = 0
        else:
            G_loss_a = 0.5 * ((1 - R ** 2 / var) * (X_n @ A)).sum() / n + lamb * np.sum(var * (X_n @ A)) / n
        G_loss_B0 = 0.5 * a * (1 - R ** 2 / var).sum(axis=0) / n + lamb * a * np.sum(var, axis=0) / n

        # back-propagate DAG constraint on gradient of A
        G_loss_A = G_loss_A + (rho_B * h_B + alpha_B) * G_h_B
        g_obj = np.concatenate((G_loss_A, -G_loss_A, G_loss_a), axis=None)
        return obj, g_obj

    n, d = X.shape
    alpha_B, rho_B, h_B = 0, 1, np.inf
    while h_B > h_tol and rho_B < rho_max:
        while rho_B < rho_max:
            sol = sopt.minimize(_func_AaB0, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            A_new, a_new = (params_new[:d * d] - params_new[d * d: 2 * d * d]).reshape([d, d]), params_new[
                2 * d * d]#, params_new[2 * d * d + 1:].reshape([1, d])
            h_new_B, _ = _h(A_new)
            if h_new_B < c * h_B:
                break
            else:
                rho_B = rho_B * s
        params_est, h_B = params_new, h_new_B
        alpha_B += rho_B * h_B
    return params_est


def update_aB0(X, X_n, params_est, bnds, lamb=0):
    def _func_aB0(params):
        # convert parameters to A, a, B0
        A = (params[:d * d] - params[d * d:2 * d * d]).reshape([d, d])
        a = params[2 * d * d]
        # B0 = params[2 * d * d + 1:].reshape([1, d])

        # compute NLL loss and pose acyclicity constraint on A
        # var = np.exp(a * X_n @ A + B0)
        # var = np.exp(a * (X_n @ A + B0))
        var = np.exp(a * (X_n @ A))
        R = X - X @ A
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        # objective = NLL loss + acyclicity penalty for B(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + lamb * np.sum(var) / n

        # Compute graident of A,B,B0, since A is fixed, the gradient for A should be 0
        G_loss_A = np.zeros([d, d])
        G_loss_a = 0.5 * ((1 - R ** 2 / var) * (X_n @ A)).sum() / n + lamb * np.sum(var * (X_n @ A)) / n
        # G_loss_B0 = 0.5 * a * (1 - R ** 2 / var).sum(axis=0) / n + lamb * a * np.sum(var, axis=0) / n

        # back-propagate DAG constraint on gradient of A
        g_obj = np.concatenate((G_loss_A, -G_loss_A, G_loss_a), axis=None)
        return obj, g_obj

    n, d = X.shape
    sol = sopt.minimize(_func_aB0, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
    params_new = sol.x
    return params_new


def nll_linear_A_a(X, A_gt, lamb=0, fix_a=True, a_fix=0, verbose=False):
    """
    a two-phrase iterative DAG learning approach
    X: input, n*d
    lamb: coefficient of uncertainty regularization
    """
    start_time = time.time()
    n, d = X.shape
    c = 0.25
    s = 10

    # centering
    X_n = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)  # X_hat
    X = X - np.mean(X, axis=0, keepdims=True)  # X_tilde

    # Initialize parameters and bounds
    A_est = np.random.normal(0, 0.0001, size=(2 * d * d))
    a_est = np.zeros([1, 1])
    # B0_est = np.zeros([1, d])
    params_est = np.concatenate((A_est, a_est), axis=None)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)] + [(None, None) for i
                                                                                                      in range(1)]
    if verbose:
        print('Update A fixing alpha and B0 to 0s.')
    params_est = update_A(X=X, X_n=X_n, params_est=params_est, bnds=bnds, c=c, s=s, lamb=lamb)
    if verbose:
        # A_est, a_est, B0_est = params2AB0(params_est, d)
        A_est = (params_est[:d * d] - params_est[d * d:2*d*d]).reshape([d, d])
        a_est = params_est[2*d*d]
        G = copy.deepcopy(A_est)
        G[np.abs(G) < 0.3] = 0
        SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
        print(f"SHD after update A is {SHD}")
        # print(f'A is: ')
        # print(G)
        print("update A, a, B0")
    if fix_a:
        params_est[2 * d * d] = a_fix
        params_est = update_AaB0(X=X, X_n=X_n, params_est=params_est, bnds=bnds, c=c, s=s, lamb=lamb, fix_a=True)
    else:
        params_est = update_AaB0(X=X, X_n=X_n, params_est=params_est, bnds=bnds, c=c, s=s, lamb=lamb, fix_a=False)
    # A_est, a_est, B0_est = params2AB0(params_est, d)
    A_est = (params_est[:d * d] - params_est[d * d:2 * d * d]).reshape([d, d])
    a_est = params_est[2 * d * d]
    if verbose:
        G = copy.deepcopy(A_est)
        G[np.abs(G) <= 0.3] = 0
        B = G * a_est
        SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
        print(f"SHD after update A, a, B0 is {SHD}")
        # print(f'A is:')
        # print(G)
    end_time = time.time()
    var_est = np.exp(a_est * (X_n @ A_est))
    nll_loss = 0.5 * (np.log(2 * np.pi * np.exp(a_est * (X_n @ A_est))) + (X - X @ A_est) ** 2 / np.exp(
        a_est * (X_n @ A_est))).sum()
    rec_loss = 0.5 / n * ((X - X @ A_est) ** 2).sum()
    return A_est, a_est, var_est, nll_loss, rec_loss, end_time - start_time


def nll_linear_A_a_alternative(X, A_gt, lamb=0, fix_a=True, a_fix=0, verbose=False):
    """
    a two-phrase iterative DAG learning approach
    X: input, n*d
    lamb: coefficient of uncertainty regularization
    """
    start_time = time.time()
    n, d = X.shape
    c = 0.25
    s = 10

    # centering
    X_n = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)  # X_hat
    X = X - np.mean(X, axis=0, keepdims=True)  # X_tilde

    # Initialize parameters and bounds
    A_est = np.random.normal(0, 0.0001, size=(2 * d * d))
    a_est = np.zeros([1, 1])
    params_est = np.concatenate((A_est, a_est), axis=None)
    # B0_est = np.zeros([1, d])
    # params_est = np.concatenate((A_est, a_est, B0_est), axis=None)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)] + [(None, None) for i
                                                                                                      in range(1)]
    losses = []
    recs = []
    losses.append(nll(X, X_n, (A_est[:d * d] - A_est[d * d:]).reshape([d, d]), a_est))
    recs.append(rec(X, (A_est[:d * d] - A_est[d * d:]).reshape([d, d])))
    l = 1
    while True:
        if verbose:
            print(f'------------------- Iteration {l}--------------------')
            print('Update A.')
        params_est[:2 * d * d] = np.random.normal(0, 0.0001, size=(2 * d * d))
        params_new = update_A(X=X, X_n=X_n, params_est=params_est, bnds=bnds, c=c, s=s, lamb=lamb)
        if verbose:
            # A_new, a_new, B0_new = params2AB0(params_new, d)
            A_new = (params_new[:d*d] - params_new[d*d:2*d*d]).reshape([d,d])
            a_new = params_new[2*d*d]
            G = copy.deepcopy(A_new)
            G[np.abs(G) < 0.3] = 0
            SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
            print(f"\tSHD after update A is {SHD}")
            # print(f'A is: ')
            # print(G)
            print(f'\trec loss: {rec(X, A_new):.4f}')
            print("update a, B0.")

        # params_new[2*d*d] = 0.1
        params_new = update_aB0(X=X, X_n=X_n, params_est=params_new, bnds=bnds, lamb=lamb)
        # A_new, a_new, B0_new = params2AB0(params_new, d)
        A_new = (params_new[:d * d] - params_new[d * d:2 * d * d]).reshape([d, d])
        a_new = params_new[2 * d * d]
        losses.append(nll(X, X_n, A_new, a_new))
        recs.append(rec(X, A_new))
        if verbose:
            G = copy.deepcopy(A_new)
            G[np.abs(G) <= 0.3] = 0
            SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
            print(f"\tSHD after update a, B0 is {SHD}.")
            print(f'\testimated alpha is {a_new:.4f}.')
            print(f'\tNll loss: {losses[-1]:.4f}')
            print(f'\trec loss: {recs[-1]:.4f}')
            # print(f'A is:')
            # print(G)
        if losses[-2] - losses[-1] < 1e-3:
            print(
                f'NLL loss from iter {l - 1} is {losses[-2]:.4f}, NLL los from iter {l} is {losses[-1]:.4f}, the difference is {losses[-2] - losses[-1]:.4f} < 1, so step.')
            break
        else:
            l += 1
            params_est = params_new
            A_est = (params_est[:d * d] - params_est[d * d:2 * d * d]).reshape([d, d])
            a_est = params_est[2 * d * d]
    # A_est, a_est, B0_est = params2AB0(params_est, d)
    # A_est = (params_est[:d * d] - params_est[d * d:2 * d * d]).reshape([d, d])
    # a_est = params_est[2 * d * d]
    end_time = time.time()
    var_est = np.exp(a_est * (X_n @ A_est))
    nll_loss = 0.5 * (np.log(2 * np.pi * np.exp(a_est * (X_n @ A_est))) + (X - X @ A_est) ** 2 / np.exp(
        a_est * (X_n @ A_est))).sum()
    rec_loss = 0.5 / n * ((X - X @ A_est) ** 2).sum()
    return A_est, a_est, var_est, nll_loss, rec_loss, end_time - start_time

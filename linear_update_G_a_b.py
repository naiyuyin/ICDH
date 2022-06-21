import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import utils as ut
import copy
import time


def nll(X, X_n, G, a, b):
    var = np.exp(X_n @ (G * tanh(b)))
    R = X - X @ (G * tanh(a))
    nll_loss = 0.5 * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
    return nll_loss


def rec(X, G, a, n):
    R = X - X @ (G * tanh(a))
    rec_loss = 0.5 / n * np.sum(R ** 2)
    return rec_loss


def tanh(x):
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)


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


def params2Gab(params, d):
    G = (params[:d*d] - params[d*d:2*d*d]).reshape([d, d])
    a = (params[2*d*d:3*d*d]).reshape([d, d])
    b = (params[3*d*d:]).reshape([d, d])
    return G, a, b


def update_G_a(X, X_n, params_est, bnds, c=0.25, s=10, lamb=0, h_tol=1e-8, rho_max=1e+16):
    def _func_G_a(params):
        # convert parameters to A, B, B0
        # G = (params[:d * d] - params[d * d:2 * d * d]).reshape([d, d])
        # a = params[2 * d * d: 3 * d * d].reshape([d, d])
        # b = params[3 * d * d:].reshape([d, d])
        G, a, b = params2Gab(params, d)

        # compute NLL loss and pose acyclicity constraint on G
        var = np.exp(X_n @ (G * tanh(b)))
        R = X - X @ (G * tanh(a))
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_G, G_h_G = _h(G)
        # objective = NLL loss + acyclicity penalty for A(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho_A * h_G * h_G + alpha_A * h_G #+ lamb * np.sum(var) / n

        # Compute graident of A,B,B0, since only updates A, gradients of B and B0 should be 0.
        G_loss_G = - tanh(a) * (X.T @ (R / var)) / n + 0.5 * tanh(b) * (X_n.T @ (1 - R ** 2 / var)) / n
        G_loss_a = np.zeros([d, d])#- G * (X.T @ (R / var)) * (1 - tanh(a) * tanh(a)) / n
        G_loss_b = np.zeros([d, d])

        # back-propagate DAG constraint on gradient of A
        G_loss_G = G_loss_G + (rho_A * h_G + alpha_A) * G_h_G
        g_obj = np.concatenate((G_loss_G, -G_loss_G, G_loss_a, G_loss_b), axis=None)
        return obj, g_obj

    n, d = X.shape
    alpha_A, rho_A, h_G = 0, 1, np.inf
    while h_G > h_tol and rho_A < rho_max:
        while rho_A < rho_max:
            sol = sopt.minimize(_func_G_a, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            # G_new, a_new, b_new = (params_new[:d * d] - params_new[d * d: 2 * d * d]).reshape([d, d]), params_new[
            #     2 * d * d: 3 * d * d].reshape([d, d]), params_new[3 * d *d:].reshape([d, d])
            G_new, a_new, b_new = params2Gab(params_new, d)
            h_new_G, _ = _h(G_new)
            if h_new_G < c * h_G:
                break
            else:
                rho_A = rho_A * s
        params_est, h_G = params_new, h_new_G
        alpha_A += rho_A * h_G
    return params_est


def update_G_a_b(X, X_n, params_est, bnds, c=0.25, s=10, lamb=0, h_tol=1e-8, rho_max=1e+16):
    def _func_G_a_b(params):
        # convert parameters to A, a, B0
        G, a, b = params2Gab(params, d)

        # compute NLL loss and pose acyclicity constraint on A
        var = np.exp(X_n @ (G * tanh(b)))
        R = X - X @ (G * tanh(a))
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_G, G_h_G = _h(G)
        # objective = NLL loss + acyclicity penalty for B(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho_B * h_G * h_G + alpha_B * h_G #+ lamb * np.sum(var) / n

        # Compute graident of A,B,B0, since A is fixed, the gradient for A should be 0
        G_loss_G = - tanh(a) * (X.T @ (R / var)) / n + 0.5 * tanh(b) * (X_n.T @ (1 - R ** 2 / var)) / n
        G_loss_a = - G * (X.T @ (R / var)) * (1 - tanh(a) ** 2) / n
        G_loss_b = 0.5 * G * ( X_n.T @ (1 - R ** 2 / var)) * (1 - tanh(b) ** 2) / n #+ lamb * a * np.sum(var, axis=0) / n

        # back-propagate DAG constraint on gradient of A
        G_loss_G = G_loss_G + (rho_B * h_G + alpha_B) * G_h_G
        g_obj = np.concatenate((G_loss_G, -G_loss_G, G_loss_a, G_loss_b), axis=None)
        return obj, g_obj

    n, d = X.shape
    alpha_B, rho_B, h_G = 0, 1, np.inf
    while h_G > h_tol and rho_B < rho_max:
        while rho_B < rho_max:
            sol = sopt.minimize(_func_G_a_b, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            G_new, a_new, b_new = params2Gab(params_new, d)
            h_new_G, _ = _h(G_new)
            if h_new_G < c * h_G:
                break
            else:
                rho_B = rho_B * s
        params_est, h_G = params_new, h_new_G
        alpha_B += rho_B * h_G
    return params_est


def update_b(X, X_n, params_est, bnds, lamb=0):
    def _func_b(params):
        # convert parameters to A, a, B0
        G, a, b = params2Gab(params, d)

        # compute NLL loss and pose acyclicity constraint on A
        var = np.exp(X_n @ (G * tanh(b)))
        R = X - X @ (G * tanh(a))
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        # objective = NLL loss + acyclicity penalty for B(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss #+ lamb * np.sum(var) / n

        # Compute graident of A,B,B0, since A is fixed, the gradient for A should be 0
        G_loss_G = np.zeros([d, d])
        G_loss_a = np.zeros([d, d])
        G_loss_b = 0.5 * G * ( X_n.T @ (1 - R ** 2 / var)) * (1 - tanh(b) ** 2) / n

        # back-propagate DAG constraint on gradient of A
        g_obj = np.concatenate((G_loss_G, -G_loss_G, G_loss_a, G_loss_b), axis=None)
        return obj, g_obj

    n, d = X.shape
    sol = sopt.minimize(_func_b, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
    params_new = sol.x
    return params_new


def update_G_a_fixVar(X, X_n, params_est, bnds, c=0.25, s=10, lamb=0, h_tol=1e-8, rho_max=1e+16):
    def _func_G_a_fixVar(params):
        # convert parameters to A, B, B0
        # G = (params[:d * d] - params[d * d:2 * d * d]).reshape([d, d])
        # a = params[2 * d * d: 3 * d * d].reshape([d, d])
        # b = params[3 * d * d:].reshape([d, d])
        G, a, b = params2Gab(params, d)

        # compute NLL loss and pose acyclicity constraint on G
        # var = np.exp(X_n @ (G * tanh(b)))
        R = X - X @ (G * tanh(a))
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_G, G_h_G = _h(G)
        # objective = NLL loss + acyclicity penalty for A(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho_A * h_G * h_G + alpha_A * h_G #+ lamb * np.sum(var) / n

        # Compute graident of A,B,B0, since only updates A, gradients of B and B0 should be 0.
        G_loss_G = - tanh(a) * (X.T @ (R / var)) / n #+ 0.5 * tanh(b) * (X_n.T @ (1 - R ** 2 / var)) / n
        G_loss_a = - G * (X.T @ (R / var)) * (1 - tanh(a) * tanh(a)) / n
        G_loss_b = np.zeros([d, d])

        # back-propagate DAG constraint on gradient of A
        G_loss_G = G_loss_G + (rho_A * h_G + alpha_A) * G_h_G
        g_obj = np.concatenate((G_loss_G, -G_loss_G, G_loss_a, G_loss_b), axis=None)
        return obj, g_obj

    n, d = X.shape
    alpha_A, rho_A, h_G = 0, 1, np.inf
    G_pre, a_pre, b_pre = params2Gab(params_est, d)
    var = np.exp(X_n @ (G_pre * tanh(b_pre)))
    while h_G > h_tol and rho_A < rho_max:
        while rho_A < rho_max:
            sol = sopt.minimize(_func_G_a_fixVar, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            G_new, a_new, b_new = params2Gab(params_new, d)
            h_new_G, _ = _h(G_new)
            if h_new_G < c * h_G:
                break
            else:
                rho_A = rho_A * s
        params_est, h_G = params_new, h_new_G
        alpha_A += rho_A * h_G
    return params_est


def nll_linear_update_G_a_b(X, A_gt, lamb, verbose):
    start_time = time.time()
    n, d = X.shape
    c = 0.25
    s = 10

    # centering
    X_n = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)  # X_hat
    X = X - np.mean(X, axis=0, keepdims=True)  # X_tilde

    # Initialize parameters and bounds
    G_est = np.random.normal(0, 0.0001, size=(2 * d * d))
    # a_est = np.random.normal(0, 0.0001, size=(d, d))
    a_est = np.ones([d, d]) # alpha
    b_est = np.zeros([d, d]) # beta
    params_est = np.concatenate((G_est, a_est, b_est), axis=None)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)] + [(0, 0) if i == j else (None, None) for _ in range(2) for i in range(d) for j in range(d)]

    if verbose:
        print('Obtain initialization of G, alpha by fixing beta -> 0.')
    params_new = update_G_a(X=X, X_n=X_n, params_est=params_est, bnds=bnds, c=c, s=s, lamb=0)
    G_new, a_new, b_new = params2Gab(params_new, d)
    if verbose:
        G = copy.deepcopy(G_new)
        G[np.abs(G) <= 0.3] = 0
        SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
        print(f"SHD after update G, alpha is {SHD}")
        print(f"Obtain initialization of beta, given G, alpha.")
    params_est = update_b(X=X, X_n=X_n, params_est=params_new, bnds=bnds, lamb=0)
    G_est, a_est, b_est = params2Gab(params_est, d)
    losses = []
    losses.append(nll(X, X_n, G_est, a_est, b_est))
    while True:
        # M step
        params_new = update_G_a_fixVar(X=X, X_n=X_n, params_est=params_est, bnds=bnds, c=c, s=s, lamb=0)
        G_new, a_new, b_new = params2Gab(params_new, d)
        if verbose:
            G = copy.deepcopy(G_new)
            G[np.abs(G) < 0.3] = 0
            SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
            print(f"SHD after update G, alpha with fixed variance is {SHD}")

        # E step
        params_new = update_b(X=X, X_n=X_n, params_est=params_new, bnds=bnds, lamb=0)
        G_new, a_new, b_new = params2Gab(params_new, d)
        nll_c = nll(X, X_n, G_new, a_new, b_new)
        if losses[-1] - nll_c > 0.1:
            losses.append(nll_c)
            params_est, G_est, a_est, b_est = params_new, G_new, a_new, b_new
        else:
            break
    var_est = np.exp(X_n @ (G_est * tanh(b_est)))
    end_time = time.time()
    return G_est, a_est, b_est, var_est, nll_c, end_time - start_time





import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import utils as ut
import copy
import time

tau = 0.2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nll(X, X_n, U, a, b):
    n, d = X.shape
    G = sigmoid(U / tau)
    G *= np.ones([d, d]) - np.eye(d)
    R = X - X @ (G * a)
    var = np.exp(X_n @ (G * b))
    nll_loss = 0.5 * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
    return nll_loss


def rec(X, U, a):
    n, d = X.shape
    G = sigmoid(U / tau)
    G *= np.ones([d, d]) - np.eye(d)
    R = X - X @ (G * a)
    rec_loss = 0.5 / n * (R ** 2).sum()
    return rec_loss


def _h(U, g):
    d = U.shape[1]
    G = sigmoid((U + g) / tau)
    G *= np.ones([d, d]) - np.eye(d)
    E = slin.expm(G)
    h = np.trace(E) - d
    G_h = E.T * G * (1 - G) * (1 / tau)
    return h, G_h


def params2Uab(params, d):
    U = params[: d * d].reshape([d, d])
    a = params[d * d: 2 * d * d].reshape([d, d])
    b = params[2 * d * d: 3 * d * d].reshape([d, d])
    return U, a, b


def update_U_a(X, X_n, params_est, bnds, c=0.25, s=10, lamb=0, h_tol=1e-10, rho_max=1e+16):
    def _func_U_a(params):
        U, a, b = params2Uab(params, d)
        g = np.random.logistic(loc=0, scale=1, size=(d, d))
        G = sigmoid((U + g) / tau)
        G *= np.ones([d, d]) - np.eye(d)

        R = X - X @ (G * a)
        var = np.exp(X_n @ (G * b))
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_U, G_h_U = _h(U, g)
        obj = loss + 0.5 * rho_A * h_U * h_U + alpha_A * h_U

        G_loss_U = G * (1 - G) * (- a * (X.T @ (R / var)) + 0.5 * b * (X_n.T @ (1 - R ** 2 / var))) / (tau * n)
        G_loss_a = - G * (X.T @ (R / var)) / n
        G_loss_b = np.zeros([d, d])

        G_loss_U = G_loss_U + (rho_A * h_U + alpha_A) * G_h_U
        g_obj = np.concatenate((G_loss_U, G_loss_a, G_loss_b), axis=None)
        return obj, g_obj

    n,d = X.shape
    alpha_A, rho_A, h_U = 0, 1, np.inf
    while h_U > h_tol and rho_A < rho_max:
        while rho_A < rho_max:
            sol = sopt.minimize(_func_U_a, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            U_new, a_new, b_new = params2Uab(params_new, d)
            # g = np.random.logistic(loc=0, scale=1, size=(d, d))
            g = np.zeros([d, d])
            h_new_U, _ = _h(U_new, g)
            if h_new_U < c * h_U:
                break
            else:
                rho_A = rho_A * s
        params_est, h_U = params_new, h_new_U
        alpha_A += rho_A * h_U
    return params_est


def update_b(X, X_n, params_est, bnds, c=0.25, s=10, lamb=0, h_tol=1e-10, rho_max=1e+16):
    def _func_b(params):
        U, a, b = params2Uab(params, d)
        g = np.random.logistic(loc=0, scale=1, size=(d, d))
        G = sigmoid((U + g) / tau)
        G *= np.ones([d, d]) - np.eye(d)

        R = X - X @ (G * a)
        var = np.exp(X_n @ (G * b))
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        obj = loss

        G_loss_U = np.zeros([d, d])
        G_loss_a = np.zeros([d, d])
        G_loss_b = 0.5 * G * ( X_n.T @ (1 - R ** 2 / var)) / n

        g_obj = np.concatenate((G_loss_U, G_loss_a, G_loss_b), axis=None)
        return obj, g_obj

    n, d = X.shape
    sol = sopt.minimize(_func_b, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
    params_new = sol.x
    return params_new


def update_U_a_fixVar(X, X_n, params_est, bnds, c=0.25, s=10, lamb=0, h_tol=1e-8, rho_max=1e+16):
    def _func_U_a_fixVar(params):
        # convert parameters to A, B, B0
        U, a, b = params2Uab(params, d)
        g = np.random.logistic(loc=0, scale=1, size=(d, d))
        G = sigmoid((U + g) / tau)
        G *= np.ones([d, d]) - np.eye(d)

        # compute NLL loss and pose acyclicity constraint on G
        R = X - X @ (G * a)
        loss = 0.5 / n * (np.log(2 * np.pi * var) + R ** 2 / var).sum()
        rec = 0.5 / n * (R ** 2).sum()
        h_U, G_h_U = _h(U, g)
        # objective = NLL loss + acyclicity penalty for A(0.5 * rho * h^2 + alpha * h) + lamb * uncertainty penalty
        obj = loss + 0.5 * rho_B * h_U * h_U + alpha_B * h_U #+ lamb * np.sum(var) / n

        # Compute graident of A,B,B0, since only updates A, gradients of B and B0 should be 0.
        G_loss_U = - G * (1 - G) * (X.T @ (R / var)) / (tau * n)
        G_loss_a = - G * (X.T @ (R / var)) / n
        G_loss_b = np.zeros([d, d])

        # back-propagate DAG constraint on gradient of A
        G_loss_U = G_loss_U + (rho_B * h_U + alpha_B) * G_h_U
        g_obj = np.concatenate((G_loss_U, G_loss_a, G_loss_b), axis=None)
        return obj, g_obj

    n, d = X.shape
    alpha_B, rho_B, h_U = 0, 1, np.inf
    U_pre, a_pre, b_pre = params2Uab(params_est, d)
    G_pre = sigmoid(U_pre / tau)
    G_pre *= np.ones([d, d]) - np.eye(d)
    var = np.exp(X_n @ (G_pre * b_pre))
    while h_U > h_tol and rho_B < rho_max:
        while rho_B < rho_max:
            sol = sopt.minimize(_func_U_a_fixVar, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            U_new, a_new, b_new = params2Uab(params_new, d)
            h_new_U, _ = _h(U_new, np.zeros([d, d]))
            if h_new_U < c * h_U:
                break
            else:
                rho_B = rho_B * s
        params_est, h_U = params_new, h_new_U
        alpha_B += rho_B * h_U
    return params_est


def nll_linear_update_binary_G_a_b(X, A_gt, lamb, verbose):
    start_time = time.time()
    n, d = X.shape
    c = 0.25
    s = 10

    # centering
    X_n = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)  # X_hat
    X = X - np.mean(X, axis=0, keepdims=True)  # X_tilde

    # Initialize parameters and bounds
    U_est = np.random.normal(0, 0.0001, size=(d, d))
    a_est = np.random.normal(0, 0.0001, size=(d, d))# np.ones([d, d]) # alpha
    b_est = np.zeros([d, d]) # beta
    params_est = np.concatenate((U_est, a_est, b_est), axis=None)
    bnds = [(0, 0) if i == j else (None, None) for _ in range(3) for i in range(d) for j in range(d)]

    if verbose:
        print('Obtain initialization of G, alpha by fixing beta -> 0.')
    params_new = update_U_a(X=X, X_n=X_n, params_est=params_est, bnds=bnds, c=c, s=5, lamb=0)
    U_new, a_new, b_new = params2Uab(params_new, d)
    if verbose:
        G = sigmoid(U_new / tau) # g = 0
        G[np.abs(G) <= 0.5] = 0
        SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
        print(f"SHD after update G, alpha is {SHD}")
        print(f"Obtain initialization of beta, given G, alpha.")
    params_est = update_b(X=X, X_n=X_n, params_est=params_new, bnds=bnds, lamb=0)
    U_est, a_est, b_est = params2Uab(params_est, d)
    losses = []
    losses.append(nll(X, X_n, U_est, a_est, b_est))
    while True:
        # M step
        params_new = update_U_a_fixVar(X=X, X_n=X_n, params_est=params_est, bnds=bnds, c=c, s=s, lamb=0)
        U_new, a_new, b_new = params2Uab(params_new, d)
        if verbose:
            G = sigmoid(U_new / tau)
            G[np.abs(G) < 0.5] = 0
            SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
            print(f"SHD after update G, alpha with fixed variance is {SHD}")

        # E step
        params_new = update_b(X=X, X_n=X_n, params_est=params_new, bnds=bnds, lamb=0)
        U_new, a_new, b_new = params2Uab(params_new, d)
        nll_c = nll(X, X_n, U_new, a_new, b_new)
        if losses[-1] - nll_c > 0.1:
            losses.append(nll_c)
            params_est, U_est, a_est, b_est = params_new, U_new, a_new, b_new
        else:
            break
    G = sigmoid(U_est / tau)
    var_est = np.exp(X_n @ (G * a_est))
    G[np.abs(G) < 0.5] = 0
    G[np.abs(G) >= 0.5] = 1
    end_time = time.time()
    return G, a_est, b_est, var_est, nll_c, end_time - start_time


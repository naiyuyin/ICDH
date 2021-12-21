import numpy as np
import scipy.optimize as sopt
import time
import copy
import utils as ut

M = 0
N = 0
epsilon = np.exp(-700)


def _h(w):
    """
    The acyclicity oncstraint introduced in Yu et al, 2019
    Input: w is the weighted adjacency matrix
    Output: h value. h(w) = 0 => w is a DAG
    """
    m = np.eye(N) + w * w / N
    e = np.linalg.matrix_power(m, N - 1)
    h = (e.T * m).sum() - N
    g_h = e.T * w * 2
    return h, g_h


def nll_expected(x, A, a, b):
    """
    Compute the new training object, the expectation of negative log-likelihood over variances
    Input:
        x is the data with dimension (M, N)
        A is the weighted adjacency matrix with dimension (N, N)
        a is the lower bounds of variance uniform distributions with dimension (1, N)
        b is the intervals of variance uniform distributions with dimension (1, N)
    Output:
        expected negative log-likelihood loss over variances.
    """
    r = x - x @ A
    return 0.5 * (((np.log(a + b + epsilon) - np.log(b + epsilon)) / (b + epsilon)) * (r ** 2)).sum() + 0.5 * M * (
            ((a + b) * np.log(a + b + epsilon) - a * np.log(a + epsilon)) / (b + epsilon)).sum() + M * N * 0.5 * (np.log(2 * np.pi) - 1)


def rec_loss(X, A):
    R = X - X @ A
    rec = 0.5 * (R ** 2).sum()
    return rec


def params2Aab(params):
    """
    Convert params into model parameters A, a, b
    Input:
        params: parameters with dimension [2*N*N + 2*N, ]
    Output:
        A: structural parameters with dimension  [N,N]
        a: variance parameters with dimension [1,N]
        b: variance parameters with dimension [1,N]
    """
    A = (params[: N * N] - params[N * N: 2 * N * N]).reshape([N, N])
    a = params[2 * N * N: 2 * N * N + N].reshape([1, N])
    b = params[2 * N * N + N:].reshape([1, N])
    return A, a, b


def update_A(X,
             params_est,
             bnds,
             c=0.25,
             s=10,
             h_tol=1e-8,
             rho_max=1e+16):
    """
    Step 1 in iterative learning approach: update structural parameters A with fixed variance parameters a,b
    Input:
        X: data with dimension [M,N]
        param_est: total parameters with dimension [2*N*N+2*N,]
        bnds: bounds for each parameters with dimension [2*N*N+2*N,]
    Output:
        param_new: updated parameters (especially for structural parameters A)
    """
    def _func_A(params):
        # convert parameters to A, a, b
        A, a, b = params2Aab(params)

        # compute NLL loss and pose acyclicity constraint on A
        R = X - X @ A
        # loss = 0.5 * (((np.log(a + b) - np.log(b)) / b) * (R ** 2)).sum() + 0.5 * M * (
        #         ((a + b) * np.log(a + b) - a * np.log(a)) / b).sum() + M * N * 0.5 * (np.log(2 * np.pi) - 1)
        # rec = 0.5 * ( R ** 2).sum()
        loss = nll_expected(X, A, a, b)
        h, G_h = _h(A)
        obj = loss + 0.5 * rho * h * h + alpha * h

        # compute gradient of A, a, b. In updateA, gradients for a, b should be 0.
        # G_loss_A = ((np.log(a + b) - np.log(a)) / b) * (- X.T @ R)
        G_loss_A = ((np.log(a + b + epsilon) - np.log(a + epsilon)) / (b + epsilon)) * (- X.T @ R)
        G_loss_a = np.zeros([1, N])
        G_loss_b = np.zeros([1, N])

        # back-propagate DAG constraint on gradient of A
        G_loss_A += (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_loss_A, -G_loss_A, G_loss_a, G_loss_b), axis=None)
        return obj, g_obj

    alpha, rho, h_A = 0, 1, np.inf
    while h_A > h_tol and rho < rho_max:
        while rho < rho_max:
            sol = sopt.minimize(_func_A, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
            params_new = sol.x
            A_new, a_new, b_new = params2Aab(params_new)
            h_new_A, _ = _h(A_new)
            if h_new_A < c * h_A:
                break
            else:
                rho *= s
        params_est, h_A = params_new, h_new_A
        alpha += rho * h_A
    return params_est


def update_a_b(X,
               params_est,
               bnds,
               lamb1 = 0,
               lamb2 = 0):
    """
        Step 2 in iterative learning approach: update variance parameters a,b with fixed structural parameters A
        Input:
            X: data with dimension [M,N]
            param_est: total parameters with dimension [2*N*N+2*N,]
            bnds: bounds for each parameters with dimension [2*N*N+2*N,]
            lamb1: coefficient for the a penalty term
            lamb2: coefficient for the b penalty term
        Output:
            param_new: updated parameters (especially for variance parameters a,b)
        """
    def _fun_a_b(params):
        # convert parameters to A, a, b
        A, a, b = params2Aab(params)

        # compute NLL loss without any acyclicity constraint
        R = X - X @ A
        # loss = 0.5 * (((np.log(a + b) - np.log(b)) / b) * (R ** 2)).sum() + 0.5 * M * (
        #         ((a + b) * np.log(a + b) - a * np.log(a)) / b).sum() + M * N * 0.5 * (np.log(2 * np.pi) - 1)
        loss = nll_expected(X, A, a, b)
        # rec = 0.5 * (R ** 2).sum()
        obj = loss + lamb1 * a.sum() + lamb2 * b.sum()

        # compute gradient of A, a, b. In updateA, gradients for a, b should be 0.
        G_loss_A = np.zeros([N, N])
        # G_loss_a = 0.5 * M * (np.log(a + b) - np.log(a)) / b - 0.5 * (R ** 2).sum(axis=0) / ((a + b) * a)
        # G_loss_b = 0.5 * M * (b + a * (np.log(a) - np.log(a + b))) / (b ** 2) + 0.5 * (R ** 2).sum(axis=0) * (
        #         1 / (a * (a + b)) + (np.log(a) - np.log(a + b)) / (b ** 2))
        G_loss_a = 0.5 * M * (np.log(a + b + epsilon) - np.log(a + epsilon)) / (b + epsilon) - 0.5 * (R ** 2).sum(axis=0) / ((a + b) * a + epsilon)
        G_loss_b = 0.5 * M * (b + a * (np.log(a + epsilon) - np.log(a + b + epsilon))) / (b ** 2 + epsilon) + 0.5 * (R ** 2).sum(axis=0) * (
                1 / (a * (a + b) + epsilon) + (np.log(a + epsilon) - np.log(a + b + epsilon)) / (b ** 2 + epsilon))

        # back-propagate DAG constraint on gradient of A
        g_obj = np.concatenate((G_loss_A, -G_loss_A, G_loss_a + lamb1, G_loss_b + lamb2), axis=None)
        return obj, g_obj

    sol = sopt.minimize(_fun_a_b, params_est, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp': 0})
    params_est = sol.x
    return params_est


def nll_linear_A_a_b(X,
                     A_gt,
                     lamb1=0,
                     lamb2=0,
                     data_type="hetero",
                     verbose=False):
    """
    Our proposed algorithm for causal discovery under heteroscedastic additive model assumption
    Input:
        X: data with dimension [M,N] -> M samples, N variables
        verbose: indicator of whether to print intermediate results
    Output:
        A_est: estimated structure before thresholding
        a_est: estimated variance uniform distributions lower bounds
        b_est: estimated variance uniform distributions' intervals.
        duration: running time
    """
    start_time = time.time()
    global M, N
    M, N = X.shape  # M is the number of samples, N is the number of nodes
    c = 0.25
    s = 10
    iteration = 1

    # centering
    # X_n = (X - np.mean(X, axis = 0, keepdims = True)) / np.std(X, axis = 0, keepdims = True)
    X = X - np.mean(X, axis=0, keepdims=True)

    # Initialize parameters and bounds
    A_est = np.random.normal(0, 0.0001, size=(2 * N * N))
    if data_type == "hetero":
        a_est = np.random.uniform(0.9, 1.0, size=(1, N))
        b_est = np.random.uniform(0.5, 1.0, size=(1, N))
    elif data_type == "homo":
        a_est = np.ones([1, N])
        b_est = np.ones([1, N])
    params_est = np.concatenate((A_est, a_est, b_est), axis=None)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(N) for j in range(N)] + \
           [(0, None) for _ in range(2) for i in range(N)]

    A_est, a_est, b_est = params2Aab(params_est)
    losses = [nll_expected(X, A_est, a_est, b_est)]

    # Iterative Approach for causal discovery
    while True:
        if verbose:
            print(f"--------------- Iteration {iteration} ---------------")
            print(" Step 1: Update structural parameter A")
        params_new = update_A(X=X, params_est=params_est, bnds=bnds, c=c, s=s)
        A_new, a_new, b_new = params2Aab(params_new)
        G = copy.deepcopy(A_new)
        G[np.abs(G) <= 0.3] = 0
        G = (G != 0).astype("int")
        SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
        print(f"SHD after update A is {SHD}")
        if verbose:
            print(" Step 2: Update variance parameter a, b")
        params_new = update_a_b(X=X, params_est=params_new, bnds=bnds, lamb1=lamb1, lamb2=lamb2)
        A_new, a_new, b_new = params2Aab(params_new)
        G = copy.deepcopy(A_new)
        G[np.abs(G) <= 0.3] = 0
        G = (G != 0).astype("int")
        SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)
        print(f"SHD after update B is {SHD}")
        losses.append(nll_expected(X, A_new, a_new, b_new))
        if verbose:
            print(f"Iteration {iteration} loss: {losses[-1]: .4f}\nIteration {iteration-1} loss: {losses[-2]: .4f}\tDecrease of the nll losses: {losses[-2] - losses[-1]:.4f}")
        iteration += 1
        if losses[-2] - losses[-1] < 10:
            if verbose:
                print("Reach convergence. Stop the iterative approach and return final estimation.")
            break
        else:
            if verbose:
                print("Go into the next iteration.")
            params_est, A_est, a_est, b_est = params_new, A_new, a_new, b_new

    end_time = time.time()
    duration = end_time - start_time
    rec = rec_loss(X, A_est)
    return A_est, a_est, b_est, duration, losses[-2], rec

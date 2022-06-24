import numpy as np
import torch
import torch.nn as nn
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
import utils as ut


class MLP(nn.Module):
    def __init__(self, dims, bias=False):
        super(MLP, self).__init()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims

        # First layer weights, W1 -> W1+, W1-
        self.W1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.W1_neg = nn.linear(d, d * dims[1], bias=bias)
        self.W1_pos.weight.bounds = self._bounds()
        self.W1_neg.weight.bounds = self._bounds()

        # Second layer weights for mean estimation W2
        self.W2 = nn.Linear(d * dims[1], d, bias=bias)

        # Second layer weights for variance estimate W3
        self.W3 = nn.Linear(d * dims[1], d, bias=bias)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):
        x = self.W1_pos(x) - self.W1_neg(x)  # [n, d * m1]
        # x = x.view(-1, self.dims[0], self.dims[1]) # [n, d, m1]
        x = torch.sigmoid(x)
        mu = self.W2(x)  # [n, d]
        # var = torch.relu(self.W3(x))
        var = torch.exp(self.W3(x))  # [n, d]
        # var = torch.exp(torch.sigmoid(self.W3(x)))
        # var = nn.Softplus(self.W3(x))
        return mu, var

    def h_func(self):
        d = self.dims[0]
        W1 = self.W1_pos.weight - self.W1_neg.weight
        W1 = W1.view(d, -1, d)
        A = torch.sum(W1 * W1, dim=1).t()
        h = trace_expm(A) - d
        # Alternative DAG constraint from yu et al, 2019
        # M = torch.eye(d) + A / d
        # E = torch.matrix_power(M, d-1)
        # h = (E.t() * M).sum() - d
        return h

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        d = self.dims[0]
        W1 = self.W1_pos.weight - self.W1_neg.weight
        W1 = W1.view(d, -1, d)
        A = torch.sum(W1 * W1, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W


def negative_log_likelihood_loss(mu, var, target):
    R = target - mu
    return 0.5 * torch.sum(torch.log(2 * np.pi * var) + R ** 2 / var)


def E_step(model: nn.Module,
           x: torch.tensor):
    model.W1_pos.requires_grad = False
    model.W1_neg.requires_grad = False
    model.W2.requires_grad = False
    model.W3.requires_grad = True
    optimizer = LBFGSBScipy(model.parameters())

    def closure():
        optimizer.zero_grad()
        x_hat, var = model(x)
        loss = negative_log_likelihood_loss(x_hat, var, x)
        loss.backward()
        return loss

    optimizer.step(closure)
    # return model


def dual_ascent_step(model, x, var, rho, alpha, h, rho_max):
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())  # check if they take no_grad
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = negative_log_likelihood_loss(x_hat, var, x)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            obj = loss + penalty
            obj.backward()
            return obj

        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def M_step(model: nn.Module,
           X: torch.tensor,
           var: torch.tensor,
           max_iter: int = 100,
           h_tol: float = 1e-8,
           rho_max: float = 1e+16):
    model.W1_pos.requires_grad = True
    model.W1_neg.requires_grad = True
    model.W2.requires_grad = True
    model.W3.requires_grad = False
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, var, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    # return model


def Nonlinear_update(model: nn.Module,
                     X: np.ndarray,
                     W_true: np.ndarray,
                     w_threshold: float = 0.3,
                     verbose: bool = True):
    X_torch = torch.from_numpy(X)
    n, d = X_torch.shape
    nlls = []

    # initial variance and W1, W2
    var_init = torch.zeros([n, d])
    M_step(model, X_torch, var_init)
    E_step(model, X_torch)
    with torch.no_grad():
        x_hat, var_est = model(X_torch)
        nll = negative_log_likelihood_loss(x_hat, var_est, X_torch).item()
        if verbose:
            w_temp = model.fc1_to_adj()
            w_temp[np.abs(w_temp) < w_threshold] = 0
            SHD, extra, missing, reverse = ut.count_accuracy(W_true, w_temp != 0)
            print(f'After initialization: NLL loss: {nll: .4f}, SHD: ({SHD}, {extra}, {missing}, {reverse})')
        nlls.append(nll)
    w_est = model.fc1_to_adj()

    # EM-updating
    while True:
        # M step
        M_step(model, X_torch, var_est)
        # E step
        E_step(model, X_torch)
        with torch.no_grad():
            x_hat, var_est = model(X_torch)
            nll = negative_log_likelihood_loss(x_hat, var_est, X_torch).item()
            if verbose:
                w_temp = model.fc1_to_adj()
                w_temp[np.abs(w_temp) < w_threshold] = 0
                SHD, extra, missing, reverse = ut.count_accuracy(W_true, w_temp != 0)
                print(f'After initialization: NLL loss: {nll: .4f}, SHD: ({SHD}, {extra}, {missing}, {reverse})')

        if nlls[-1] - nll < 1e-1:
            break
        else:
            nlls.append(nll)
            w_est = model.fc1_to_adj()

    w_est[np.abs(w_est) < w_threshold] = 0
    return w_est, nlls


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # generate synthetic data
    ut.set_random_seed(123)
    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mlp'
    B_true = ut.simulate_dag(d, s0, graph_type)
    np.save('W_true.csv', B_true, delimiter=',')
    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    np.save('X.csv', X, delimiter=',')

    model = MLP(dims=[d, 10, 1], bias=False)
    A_est = Nonlinear_update(model, X)
    assert ut.is_dag(A_est)
    np.savetxt('W_est.csv', A_est, delimiter=',')
    SHD, _, _, _ = ut.count_accuracy(B_true, A_est != 0)
    print(SHD)


if __name__ == '__main__':
    main()

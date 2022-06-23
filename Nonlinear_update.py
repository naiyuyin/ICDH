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
        self.W1_pos = nn.Linear(d, d*dims[1], bias=bias)
        self.W1_neg = nn.linear(d, d*dims[1], bias=bias)
        self.W1_pos.weight.bounds = self._bounds()
        self.W1_neg.weight.bounds = self._bounds()

        # Second layer weights for mean estimation W2
        self.W2 = nn.Linear(d*dims[1], d, bias=bias)

        # Second layer weights for variance estimate W3
        self.W3 = nn.Linear(d*dims[1], d, bias=bias)

    def _bounds(self):
        d = self.dims[0]
        bounds =  []
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
        x = self.W1_pos(x) - self.W1_neg(x) # [n, d * m1]
        # x = x.view(-1, self.dims[0], self.dims[1]) # [n, d, m1]
        x = torch.sigmoid(x)
        mu = self.W2(x) # [n, d]
        # var = torch.relu(self.W3(x))
        var = torch.exp(self.W3(x)) # [n, d]
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
        W = W.cpu().detach().num()
        return W


def negative_log_likelihood_loss(mu, var, target):
    R = target - mu
    nll_loss = 0.5 * torch.sum(torch.log(2 * np.pi * var) + R ** 2 / var)
    return nll_loss


def E_step(model, X):
    return model


def M_step(model: nn.Module,
           X: np.ndarray,
           max_iter: int = 100,
           h_tol: float = 1e-8,
           rho_max: float = 1e+16):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    return model


def Nonlinear_update(model: nn.Module,
                     X: np.ndarray,
                     w_threshold: float = 0.3):
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return A_est


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
    SHD, _, _, _ = ut.count_accuracy(B_true, A_est !=0)
    print(SHD)


if __name__=='__main__':
    main()




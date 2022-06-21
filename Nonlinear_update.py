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
        return mu, var

    def h_func(self):
        d = self.dims[0]
        W1 = self.W1_pos.weight - self.W1_neg.weight
        W1 = W1.view(d, -1, d)
        A = torch.sum(W1 * W1, dim=1).t()
        h = trace_expm(A) - d
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


def Nonlinear_update(model, X):
    return A_est


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    model = MLP(dims=[d, 10, 1], bias=False)
    A_est = Nonlinear_update(model, X)
    assert ut.is_dag(A_est)


if __name__=='__main__':
    main()




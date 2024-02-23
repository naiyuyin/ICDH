import numpy as np
import torch
import torch.nn as nn
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
from locally_connected import LocallyConnected
import utils as ut
import os
from torch.nn import functional as F
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MLP(nn.Module):
    def __init__(self, dims, device, bias=False):
        super(MLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims

        # First layer weights, W1 -> W1+, W1-
        self.W1_pos = nn.Linear(d, d * dims[1], bias=bias).to(device)
        self.W1_neg = nn.Linear(d, d * dims[1], bias=bias).to(device)
        self.W1_pos.weight.bounds = self._bounds()
        self.W1_neg.weight.bounds = self._bounds()

        # Second layer weights for mean estimation W2
        self.W2 = LocallyConnected(d, dims[1], 1, bias=bias).to(device)
        self.W2.weight.data[:] = torch.from_numpy(np.random.randn(d, dims[1], 1)).to(device)

        # Second layer weights for variance estimate W3
        self.W3 = LocallyConnected(d, dims[1], 1, bias=bias).to(device)
        self.W3.weight.data[:] = torch.from_numpy(np.random.randn(d, dims[1], 1)).to(device)
        self.W4 = nn.Parameter(torch.randn(1,)).to(device)
        self.acfun = nn.Softplus()
        self.tanh = nn.Tanh()

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
        x = x.view(-1, self.dims[0], self.dims[1]) # [n, d, m1]
        h = torch.sigmoid(x) # [n, d, m1]
        mu = self.W2(h) # [n, d, m2 = 1]
        mu = mu.squeeze(dim=2)  # [n, d]

        var = F.relu(self.W3(h)) + torch.sigmoid(self.W4)
        # Other possible functions for estimating variances.
        # var = torch.exp(self.W3(x))  # [n, d, m2 = 1]
        # var = torch.exp(torch.sigmoid(self.W3(x)))
        # var = self.acfun(self.W3(x))
        # var = F.relu(torch.nn.Tanh(self.W3(h)))
        var = var.squeeze(dim=2) # [n, d]
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

    def l2_reg(self):
        reg = 0.
        W1_weight = self.W1_pos.weight - self.W1_neg.weight  # [j * m1, i]
        reg += torch.sum(W1_weight ** 2)
        reg += torch.sum(self.W2.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.W1_pos.weight + self.W1_neg.weight)
        return reg

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
    n = target.shape[0]
    R = target - mu
    return 0.5 / n * torch.sum(torch.log(2 * np.pi * var) + R ** 2 / var)


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def A_step(model: nn.Module,
           x: torch.tensor):
    # This is the Phase-I step, where we estimate the optimal values for variances.
    model.W1_pos.weight.requires_grad = False
    model.W1_neg.weight.requires_grad = False
    model.W2.weight.requires_grad = False
    model.W3.weight.requires_grad = True
    optimizer = LBFGSBScipy(model.parameters())

    def closure():
        optimizer.zero_grad()
        x_hat, var = model(x)
        loss = negative_log_likelihood_loss(x_hat, var, x)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        x_hat, var = model(x)
        loss = negative_log_likelihood_loss(x_hat, var, x).item()
        print(f'NLL loss: {loss: .4f}.')


def dual_ascent_step(model, x, var, lamb1, lamb2, rho, alpha, h, rho_max):
    # lamb1 and lamb2 are the coefficients for the L1 and L2 regualrization. The values for these two hypterparameters need to be tuned.
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = negative_log_likelihood_loss(x_hat, var, x)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lamb2 * model.l2_reg()
            l1_reg = lamb1 * model.fc1_l1_reg()
            obj = loss + penalty + l2_reg + l1_reg
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


def B_step(model: nn.Module,
           X: torch.tensor,
           var: torch.tensor,
           lamb1: float,
           lamb2: float,
           max_iter: int = 100,
           h_tol: float = 1e-8,
           rho_max: float = 1e+16):
    # This is the Phase-II step, where we optimize over the structural parameters using fixed variances.
    model.W1_pos.weight.requires_grad = True
    model.W1_neg.weight.requires_grad = True
    model.W2.weight.requires_grad = True
    model.W3.weight.requires_grad = False
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, var, lamb1, lamb2, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break


def ICDH(model,
         X,
         lamb1,
         lamb2,
         device,
         W_true,
         w_threshold=0.3,
         tol=1e-2,
         verbose=True):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    X_torch = torch.from_numpy(X).to(device)
    n, d = X_torch.shape
    var_init = torch.ones([n, d]).to(device)
    nlls = []
    recs = []

    # initial variance and W1, W2
    B_step(model=model, X=X_torch, var=var_init, lamb1=lamb1, lamb2=lamb2)
    A_step(model=model, x=X_torch)
    with torch.no_grad():
        x_hat, var_est = model(X_torch)
        nll = negative_log_likelihood_loss(x_hat, var_est, X_torch).item()
        rec = squared_loss(x_hat, X_torch).item()
        if verbose:
            w_temp = model.fc1_to_adj()
            w_temp[np.abs(w_temp) < w_threshold] = 0
            if ut.is_dag(w_temp):
                SHD, extra, missing, reverse = ut.count_accuracy(W_true, w_temp != 0)
                print(f'After initialization: NLL loss: {nll: .4f}, rec loss: {rec: .4f}, SHD: ({SHD}, {extra}, {missing}, {reverse})')
            else:
                pass
        nlls.append(nll)
        recs.append(rec)
    w_est = model.fc1_to_adj()

    # Iterative updates
    while True:
        # Phase-II
        B_step(model, X_torch, var_est, lamb1, lamb2)
        # Phase-I
        A_step(model, X_torch)
        with torch.no_grad():
            x_hat, var_est = model(X_torch)
            nll = negative_log_likelihood_loss(x_hat, var_est, X_torch).item()
            rec = squared_loss(x_hat, X_torch).item()
            if verbose:
                w_temp = model.fc1_to_adj()
                w_temp[np.abs(w_temp) < w_threshold] = 0
                SHD, extra, missing, reverse = ut.count_accuracy(W_true, w_temp != 0)
                print(f'After initialization: NLL loss: {nll: .4f}, rec loss: {rec: .4f}, SHD: ({SHD}, {extra}, {missing}, {reverse})')

        if nlls[-1] - nll < tol:
            break
        else:
            nlls.append(nll)
            w_est = model.fc1_to_adj()

    # depends on whether you want the thresholded results or not
    # w_est[np.abs(w_est) < w_threshold] = 0
    return w_est, nlls


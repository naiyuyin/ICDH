import numpy as np
import torch
import torch.nn as nn
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
from locally_connected import LocallyConnected
import utils as ut
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MLP(nn.Module):
    def __init__(self, dims, bias=False):
        super(MLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims

        # First layer weights, W1 -> W1+, W1-
        self.W1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.W1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.W1_pos.weight.bounds = self._bounds()
        self.W1_neg.weight.bounds = self._bounds()

        # Second layer weights for mean estimation W2
        self.W2 = LocallyConnected(d, dims[1], 1, bias=bias)
        self.W2.weight.data[:] = torch.from_numpy(np.random.randn(d, dims[1], 1))

        # Second layer weights for variance estimate W3
        self.W3 = LocallyConnected(d, dims[1], 1, bias=bias)
        self.W3.weight.data[:] = torch.from_numpy(np.random.randn(d, dims[1], 1))
        self.acfun = nn.Softplus()

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
        x = torch.sigmoid(x) # [n, d, m1]
        mu = self.W2(x) # [n, d, m2 = 1]
        mu = mu.squeeze(dim=2)  # [n, d]

        # var = torch.exp(self.W3(x))  # [n, d, m2 = 1]
        # var = torch.exp(torch.sigmoid(self.W3(x)))
        var = self.acfun(self.W3(x))
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
    R = target - mu
    return 0.5 * torch.sum(torch.log(2 * np.pi * var) + R ** 2 / var)


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def E_step(model: nn.Module,
           x: torch.tensor):
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
    # return model


def dual_ascent_step(model, x, var, lamb1, lamb2, rho, alpha, h, rho_max):
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())  # check if they take no_grad
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = negative_log_likelihood_loss(x_hat, var, x)
            # loss = squared_loss(x_hat, x)
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


def M_step(model: nn.Module,
           X: torch.tensor,
           var: torch.tensor,
           lamb1: float,
           lamb2: float,
           max_iter: int = 100,
           h_tol: float = 1e-8,
           rho_max: float = 1e+16):
    model.W1_pos.weight.requires_grad = True
    model.W1_neg.weight.requires_grad = True
    model.W2.weight.requires_grad = True
    model.W3.weight.requires_grad = False
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, var, lamb1, lamb2, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    # return model


def Nonlinear_update(model,
                     X,
                     lamb1,
                     lamb2,
                     device,
                     W_true,
                     w_threshold=0.3,
                     verbose=True):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    X_torch = torch.from_numpy(X).to(device)
    n, d = X_torch.shape
    var_init = torch.ones([n, d]).to(device)
    nlls = []

    # initial variance and W1, W2
    M_step(model=model, X=X_torch, var=var_init, lamb1=lamb1, lamb2=lamb2)
    E_step(model=model, x=X_torch)
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
        M_step(model, X_torch, var_est, lamb1, lamb2)
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

    # w_est[np.abs(w_est) < w_threshold] = 0
    return w_est, nlls


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # generate synthetic data
    # ut.set_random_seed(123)
    # n, d, s0, graph_type, sem_type = 1000, 5, 9, 'ER', 'mlp'
    # B_true = ut.simulate_dag(d, s0, graph_type)
    # np.savetxt('W_true.csv', B_true, delimiter=',')
    # X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    # np.savetxt('X.csv', X, delimiter=',')

    # load data
    X = np.loadtxt('X.csv', delimiter=',')
    W_true = np.loadtxt('W_true.csv', delimiter=',')
    n, d = X.shape

    model = MLP(dims=[d, 10, 1], bias=False)
    A_est, nlls = Nonlinear_update(model=model, X=X, lamb1=0.05, lamb2=0.05, W_true=W_true)
    # M_step(model=model, X=torch.from_numpy(X), var=torch.ones([n, d]), lamb1=0.05, lamb2=0.05)
    # torch.save({'model_state_dict': model.state_dict()}, 'model_init.pt')
    # checkpoint = torch.load('model_init.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # #
    # E_step(model, torch.from_numpy(X))
    # A_est = model.fc1_to_adj()
    # A_est[A_est < 0.3] = 0
    assert ut.is_dag(A_est)
    SHD, extra, missing, reverse = ut.count_accuracy(W_true, A_est != 0)
    print(f"SHD: {SHD}, extra: {extra}, missing: {missing}, reverse: {reverse}.")


if __name__ == '__main__':
    main()

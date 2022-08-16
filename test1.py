import numpy as np
from linear_update_A_B import nll_linear_A_B
from test_linear_update import nll_linear_AB


def main():
    idx = 2
    X = np.loadtxt(f'data/hetero_old/ER1_d5/data/X_{idx}.txt', delimiter=',')
    A_gt = np.loadtxt(f'data/hetero_old/ER1_d5/graph/A_{idx}.txt', delimiter=',')
    print(A_gt)

    A_est, B_est, B0_est, var_est, nll, rec, time = nll_linear_A_B(X=X, A_gt=A_gt, lamb=0, verbose=True)
    A_est[np.abs(A_est) < 0.3] = 0
    print(A_est)
    print(np.mean(var_est))
    print(f'Nll: {nll}, Rec: {rec}, Time: {time}.')

    A_est, B_est, B0_est, var_est, nll, rec, time = nll_linear_AB(X=X, A_gt=A_gt, lamb=0, verbose=True)
    A_est[np.abs(A_est) < 0.3] = 0
    print(A_est)
    print(np.mean(var_est))
    print(f'Nll: {nll}, Rec: {rec}, Time: {time}.')


if __name__=="__main__":
    main()
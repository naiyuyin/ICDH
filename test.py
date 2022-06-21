import numpy as np
from linear_update_A import nll_linear_A
from linear_update_A_a import nll_linear_A_a_alternative as nll_linear_A_a_alter
from linear_update_A_a import nll_linear_A_a as nll_linear_A_a
from linear_update_G_a_b import nll_linear_update_G_a_b
# from Linear_update_binary_G_a_b import nll_linear_update_binary_G_a_b
from linear_update_A_B import nll_linear_A_B as nll_linear
import utils as ut
import copy
import argparse
from notears.linear import notears_linear
import os


def main():
    data_type = 'homo'
    graph_setting = 'ER2_d10'
    lamb = 0
    idx = 1

    # load data
    A_gt = np.loadtxt(f'data/{data_type}/{graph_setting}/graph/A_{idx}.txt', delimiter=",")
    # a = np.loadtxt(f'data/{data_type}/{graph_setting}/graph/a_{idx}.txt', delimiter=",")
    X = np.loadtxt(f'data/{data_type}/{graph_setting}/data/X_{idx}.txt', delimiter=",")
    # print(f'ground-truth alpha: {a:.4f}')
    # print('ground-truth A: ')
    # print(A_gt)

    # Our method
    # A_est, a_est, var_est, nll, rec, time = nll_linear_A_a(X=X, A_gt=A_gt, lamb=lamb, fix_a=False, a_fix=None, verbose=True)
    # A_est, a_est, var_est, nll, rec, time = nll_linear_A_a_alter(X=X, A_gt=A_gt, lamb=lamb, verbose=True)
    G_est, a_est, b_est, var_est, nll = nll_linear_update_G_a_b(X=X, A_gt=A_gt, lamb=lamb, verbose=True)
    # U_est, a_est, b_est, var_est, nll = nll_linear_update_binary_G_a_b(X=X, A_gt=A_gt, lamb=0, verbose=True)
    # print(f'NLL loss: {nll: .4f}')
    # print(f'rec loss: {rec: .4f}')
    # print(f'variance mean: {np.mean(var_est): .4f}')
    # print(f'estimated alpha is: {a_est: .4f}')
    # print('estimated A: ')
    # print(A_est)
    # print('estimated B: ')
    # print(a_est * A_est)
    # print('estimated B0: ')
    # print(A0_est)
    # np.savetxt('var_est.txt', var_est, delimiter=",")

    # A_est, B_est, B0_est, var_est, nll, rec, time = nll_linear(X=X, A_gt=A_gt, lamb=lamb, verbose=True)
    # print(f'NLL loss: {nll: .4f}')
    # print(f'rec loss: {rec: .4f}')
    # print(f'variance mean: {np.mean(var_est): .4f}')
    # print('estimated A: ')
    # print(A_est)
    # print('estimated B: ')
    # print(B_est)
    # print('estimated B0: ')
    # print(B0_est)
    # np.savetxt('var_est.txt', var_est, delimiter=",")

    # ut.set_random_seed(123)
    # A_est = notears_linear(X=X, lambda1=0, loss_type="l2")
    # print('estimated A: ')
    # print(A_est)
    # X_c = X - np.mean(X, axis=0, keepdims=True)
    # n,d = X_c.shape
    # print(f'MSE loss: {0.5 / n * ((X_c - X_c @ A_est) ** 2).sum(): .4f}')

    # evaluate
    n, d = X.shape
    G = copy.deepcopy(G_est)
    G[np.abs(G) <= 0.3] = 0
    G = (G != 0).astype("int")
    SHD, extra, missing, reverse = ut.count_accuracy(A_gt, G != 0)  # SHD with thres as 0.3
    print(f'SHD: {SHD}, extra: {extra}, missing: {missing}, reverse: {reverse}')


def main_all(args):
    data_type = args.data_type
    degree = args.s0
    d = args.num_size
    graph_type = args.graph_type
    lamb = args.lamb_a
    res = np.zeros([10, 5])
    np.random.seed(42)
    res_dir = f"results/{data_type}/{graph_type}{degree}_d{d}"
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    np.savetxt(os.path.join(res_dir, "nll_res.txt"), res, delimiter=",")

    for idx in range(4, 11):
        print(f"--------------------------------graph {idx}---------------------------------------")
        res = np.loadtxt(os.path.join(res_dir, "nll_res.txt"), delimiter=",")
        # load data
        A_gt = np.loadtxt(f'data/{data_type}/{graph_type}{degree}_d{d}/graph/A_{idx}.txt', delimiter=",")
        X = np.loadtxt(f'data/{data_type}/{graph_type}{degree}_d{d}/data/X_{idx}.txt', delimiter=",")

        # Our method
        G_est, a_est, b_est, var_est, nll, time = nll_linear_update_G_a_b(X=X, A_gt=A_gt, lamb=lamb, verbose=True)
        np.savetxt(os.path.join(res_dir, f'G_est_{idx}.txt'), G_est, delimiter=",")
        np.savetxt(os.path.join(res_dir, f'a_est_{idx}.txt'), a_est, delimiter=",")
        np.savetxt(os.path.join(res_dir, f'b_est_{idx}.txt'), b_est, delimiter=",")
        np.savetxt(os.path.join(res_dir, f'var_est_{idx}.txt'), var_est, delimiter=",")

        # evaluate
        G = copy.deepcopy(G_est)
        G[np.abs(G) <= 0.3] = 0
        G = (G != 0).astype("int")
        SHD, extra, missing, reverse = ut.count_accuracy(A_gt, G != 0)  # SHD with thres as 0.3
        res[idx-1, :] = SHD, extra, missing, reverse, time
        np.savetxt(os.path.join(res_dir, "nll_res.txt"), res, delimiter=",")
        print(f'SHD: {SHD}, extra: {extra}, missing: {missing}, reverse: {reverse}')
    print(f"SHD: {np.mean(res, axis = 0)[0]} +/ {np.std(res, axis = 0)[0]:.2f}")
    np.savetxt(os.path.join(res_dir, "nll_res.txt"), res, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000,
                        help="sample size of the generated data")
    parser.add_argument('--graph_type', type=str, default='ER',
                        choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=2, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=10,
                        help="variable size to be generated")
    parser.add_argument('--data_type', type=str, default='hetero', choices=['homo', 'hetero', 'homo_nv', 'hetero_old'], help='data type')
    # parser.add_argument('--formulation', type=int, default=1, choices=[1, 2], help='formuation')

    # parameter tuning
    parser.add_argument('--lamb_a', type=float, default=0.1,
                        help="The coefficient of l1 regularization on variance distribution lower bounds a.")
    parser.add_argument('--lamb_b', type=float, default=0.0,
                        help="The coefficient of l1 regularization on variance distribution interval b.")
    parser.add_argument('--cuda_device', type=str, default="0", choices=["0", "1", "2", "3", "4", "5", "6"])

    # evaluation parameters
    parser.add_argument('--thres_min', type=float, default=0.1, help="The minimum value of threshold.")
    parser.add_argument('--thres_max', type=float, default=0.75, help="The maximum value of threshold.")

    args = parser.parse_args()
    main_all(args)
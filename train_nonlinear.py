import numpy as np
import torch
import torch.nn as nn
from notears.nonlinear import notears_nonlinear
from notears.nonlinear import NotearsMLP
import utils as ut
import time as t
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main_notears(args):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # X = np.loadtxt('X.csv', delimiter=',')
    # W_true = np.loadtxt('W_true.csv', delimiter=',')
    # X = np.loadtxt("/Users/naiyuyin/Documents/Research/data/Sachs_data/sachs_X.txt", delimiter=",")
    # W_true = np.loadtxt("/Users/naiyuyin/Documents/Research/data/Sachs_data/sachs_W.txt", delimiter=",")
    # n, d = X.shape
    # model = NotearsMLP(dims=[d, 32, 1], bias=True)
    # W_est = notears_nonlinear(model, X, lambda1=2, lambda2=2) # no l1 and l2 constraint version
    # np.savetxt("A_notears.txt", X, delimiter=",")
    # assert ut.is_dag(W_est)
    # SHD, extra, missing, reverse = ut.count_accuracy(W_true, W_est != 0)
    # print(f'SHD {SHD}, extra {extra}, missing {missing}, reverse {reverse}.')


    res = np.zeros([10, 5])
    np.savetxt(f'results/nonlinear_{args.data_type}_{args.graph_type}{args.s0}_d{args.num_size}.csv', res, delimiter=',')
    res_path = f'results/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}'
    if not os.path.isdir(f'results/{args.data_type}'):
        os.mkdir(f'results/{args.data_type}')
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    for s in range(10):
        # load data
        print("\n--------------------------------------------------")
        res = np.loadtxt(f'results/nonlinear_{args.data_type}_{args.graph_type}{args.s0}_d{args.num_size}.csv', delimiter=',')
        X = np.loadtxt(f'data/nonlinear/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}/data/X_{s+1}.csv', delimiter=',')
        W_true = np.loadtxt(f'data/nonlinear/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}/graph/W_{s+1}.csv', delimiter=',')
        n, d = X.shape

        # notears nonlinear
        start_time = t.time()
        model = NotearsMLP(dims=[d, 10, 1], bias=True)
        W_est = notears_nonlinear(model, X, lambda1=args.lamb1, lambda2=args.lamb2) # no l1 and l2 constraint version
        end_time = t.time()
        time = end_time - start_time
        np.savetxt(res_path + f'/W_{s+1}.csv', W_est, delimiter=',')
        W_est[W_est < 0.3] = 0
        assert ut.is_dag(W_est)
        SHD, extra, missing, reverse = ut.count_accuracy(W_true, W_est != 0)
        res[s,:] = SHD, extra, missing, reverse, time
        print(f'Graph {s+1}: SHD {SHD}, extra {extra}, missing {missing}, reverse {reverse}, time {time: .4f}.')
        print("--------------------------------------------------\n")
        np.savetxt(f'results/nonlinear_{args.data_type}_{args.graph_type}{args.s0}_d{args.num_size}.csv', res, delimiter=',')
    mean_res = np.mean(res, axis=0)
    std_res = np.std(res, axis=0)
    print(f'SHD: {mean_res[0]:.1f} +/- {std_res[0]:.4f}\nTime: {mean_res[4]:.4f} +/- {std_res[4]:.4f}')


def main_syn(args):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # set up random seed
    # ut.set_random_seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)

    # generate folder for the generated data
    print(f"Generate {args.data_type} data with {args.num_size} nodes, {args.sample_size} samples from {args.graph_type}{args.s0} graph.")
    path = f"data/nonlinear/{args.data_type}"
    file_dir = f"data/nonlinear/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}"
    data_dir = f"data/nonlinear/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}/data"
    graph_dir = f"data/nonlinear/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}/graph"

    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
        os.mkdir(data_dir)
        os.mkdir(graph_dir)

    n, d, s0, graph_type, sem_type = args.sample_size, args.num_size, args.num_size * args.s0, args.graph_type, args.sem
    for s in range(10):
        W_true = ut.simulate_dag(d, s0, graph_type)
        np.savetxt(graph_dir + f"/W_{s+1}.csv", W_true, delimiter=',')
        X = ut.simulate_nonlinear_sem(W_true, n, sem_type)
        np.savetxt(data_dir + f"/X_{s+1}.csv", X, delimiter=',')
    # W_true = ut.simulate_dag(d, s0, graph_type)
    # np.savetxt(f"W.csv", W_true, delimiter=',')
    # X = ut.simulate_nonlinear_sem(W_true, n, sem_type)
    # np.savetxt(f"X.csv", X, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000, help="sample size of the generated data")
    parser.add_argument('--graph_type', type=str, default='ER', choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=2, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=10, help="variable size to be generated")
    parser.add_argument('--data_type', type=str, default='homo_ev', choices=['homo_ev', 'hetero', 'homo_nv'],help='data type')
    parser.add_argument('--sem', type=str, default='mlp', choices=['mlp', 'gp'], help='Types of SEM model')
    parser.add_argument('--lamb1', type=float, default=0.01, help="The coefficient of l1 regularization on parameters.")
    parser.add_argument('--lamb2', type=float, default=0.01, help="The coefficient of l2 regularization on parameters.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seeds.")
    args = parser.parse_args()
    # main_syn(args)
    main_notears(args)

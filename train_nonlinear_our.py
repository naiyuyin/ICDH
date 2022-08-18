import numpy as np
import torch
import torch.nn as nn
from notears.nonlinear import notears_nonlinear
from notears.nonlinear import NotearsMLP
from Nonlinear_update import MLP
from Nonlinear_update import Nonlinear_update
import utils as ut
import time as t
import argparse
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main_our(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    torch.manual_seed(args.random_seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)


    res = np.zeros([10, 5])
    np.savetxt(f'results/our_nonlinear_{args.data_type}_{args.graph_type}{args.s0}_d{args.num_size}.csv', res, delimiter=',')
    res_path = f'results/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}'
    if not os.path.isdir(f'results/{args.data_type}'):
        os.mkdir(f'results/{args.data_type}')
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    for s in range(10):
        # load data
        print("\n--------------------------------------------------")
        res = np.loadtxt(f'results/our_nonlinear_{args.data_type}_{args.graph_type}{args.s0}_d{args.num_size}.csv', delimiter=',')
        X = np.loadtxt(f'data/nonlinear/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}/data/X_{s+1}.csv', delimiter=',')
        W_true = np.loadtxt(f'data/nonlinear/{args.data_type}/{args.graph_type}{args.s0}_d{args.num_size}/graph/W_{s+1}.csv', delimiter=',')
        n, d = X.shape

        # notears nonlinear
        start_time = t.time()
        model = MLP(dims=[d, 10, 1], bias=True).to(device=device)
        W_est = Nonlinear_update(model=model, X=X, lamb1=args.lamb1, lamb2=args.lamb2, device=device, W_true=W_true) # no l1 and l2 constraint version
        end_time = t.time()
        time = end_time - start_time
        np.savetxt(res_path + f'/W_nll_{s+1}.csv', W_est, delimiter=',')
        W_est[W_est < 0.3] = 0
        assert ut.is_dag(W_est)
        SHD, extra, missing, reverse = ut.count_accuracy(W_true, W_est != 0)
        res[s,:] = SHD, extra, missing, reverse, time
        print(f'Graph {s+1}: SHD {SHD}, extra {extra}, missing {missing}, reverse {reverse}, time {time: .4f}.')
        print("--------------------------------------------------\n")
        np.savetxt(f'results/our_nonlinear_{args.data_type}_{args.graph_type}{args.s0}_d{args.num_size}.csv', res, delimiter=',')
    mean_res = np.mean(res, axis=0)
    std_res = np.std(res, axis=0)
    print(f'SHD: {mean_res[0]:.1f} +/- {std_res[0]:.4f}\nTime: {mean_res[4]:.4f} +/- {std_res[4]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000, help="sample size of the generated data")
    parser.add_argument('--graph_type', type=str, default='ER', choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=1, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=10, help="variable size to be generated")
    parser.add_argument('--data_type', type=str, default='homo_ev', choices=['homo_ev', 'hetero', 'homo_nv'],help='data type')
    parser.add_argument('--sem', type=str, default='mlp', choices=['mlp', 'gp'], help='Types of SEM model')
    parser.add_argument('--lamb1', type=float, default=0.01, help="The coefficient of l1 regularization on parameters.")
    parser.add_argument('--lamb2', type=float, default=0.01, help="The coefficient of l2 regularization on parameters.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seeds.")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cuda_device', type=str, default="1", choices=["0", "1", "2", "3", "4", "5"])
    args = parser.parse_args()
    # main_syn(args)
    # main_notears(args)
    main_our(args)
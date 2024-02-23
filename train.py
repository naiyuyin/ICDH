import numpy as np
import torch
import torch.nn as nn
from ICDH import MLP
from ICDH import ICDH
import utils as ut
import time as t
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def train_ICDH(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    torch.manual_seed(args.random_seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # generate synthetic data
    print(f"Generate {args.sample_size} nonlinear {args.data_type} data from {args.graph_type}{args.s0} graphs with {args.num_size} variables.")
    n, d, s0, graph_type, sem_type = args.sample_size, args.num_size, args.num_size * args.s0, args.graph_type, args.sem
    W_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt('W_true.csv', W_true, delimiter=',')
    X = ut.simulate_nonlinear_sem_hetero(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    # run ICDH
    print(f"Run our ICDH method on the generated data.")
    model = MLP(dims=[d, 10, 1], bias=False)
    start_time = t.time()
    A_est, nlls = ICDH(model=model, X=X, lamb1=args.lamb1, lamb2=args.lamb2, W_true=W_true)
    A_est[A_est < 0.3] = 0
    end_time = t.time()
    time = end_time - start_time
    assert ut.is_dag(A_est)
    SHD, extra, missing, reverse = ut.count_accuracy(W_true, A_est != 0)
    print(f"SHD: {SHD} (extra: {extra}, missing: {missing}, reverse: {reverse}), Runtime: {time:.2f}s.")
    np.savetxt(f'A_ICDH.csv', A_est, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000, help="sample size of the generated data")
    parser.add_argument('--graph_type', type=str, default='ER', choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=2, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=5, help="variable size to be generated")
    parser.add_argument('--data_type', type=str, default='hetero_nonlinear', choices=['homo_ev_nonlinear', 'hetero_nonlinear', 'homo_nv_nonlinear'],help='data type')
    parser.add_argument('--sem', type=str, default='mlp', choices=['mlp', 'gp'], help='Types of SEM model')
    parser.add_argument('--lamb1', type=float, default=0.05, help="The coefficient of l1 regularization on parameters.")
    parser.add_argument('--lamb2', type=float, default=0.05, help="The coefficient of l2 regularization on parameters.")
    parser.add_argument('--random_seed', type=int, default=123, help="Random seeds.")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cuda_device', type=str, default="1", choices=["0", "1", "2", "3", "4", "5"])
    args = parser.parse_args()
    train_ICDH(args)

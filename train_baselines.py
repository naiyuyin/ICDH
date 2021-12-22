import numpy as np
import pandas as pd
from notears.linear import notears_linear
import time
import utils as ut
import copy
import matplotlib.pyplot as plt
import argparse
import os


def train_synthetic_notears(args):
    res = np.zeros([10, 6])

    # specify parameters
    n = args.sample_size
    thresholds = np.linspace(args.thres_min, args.thres_max, endpoint=True, num=1000)
    d = args.num_size
    graph_type = args.graph_type
    s0 = args.s0
    np.random.seed(123)
    data_type = args.data_type

    res_dir = f"results/{data_type}/{graph_type}{s0}_d{d}_n{n}"
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    np.savetxt(os.path.join(res_dir, "notears_res.txt"), res, delimiter=",")

    for iter in range(10):
        print(f"--------------------------- graph {iter + 1} ---------------------------")
        res = np.loadtxt(os.path.join(res_dir, "notears_res.txt"), delimiter=",")
        X = np.loadtxt(os.path.join(f"data/{data_type}/{graph_type}{s0}_d{d}/data", f"X_{iter + 1}.txt"), delimiter=",")
        X = X[:n, :]
        A_gt = np.loadtxt(os.path.join(f"data/{data_type}/{graph_type}{s0}_d{d}/graph", f"G_{iter + 1}.txt"),
                          delimiter=",")

        np.random.seed(123)
        start_time = time.time()
        A_notears = notears_linear(X=X, lambda1=0, loss_type="l2")
        end_time = time.time()
        res[iter, 1] = end_time - start_time
        np.savetxt(os.path.join(res_dir, f"A_notears_{iter + 1}.txt"), A_notears, delimiter=",")

        G = copy.deepcopy(A_notears)
        G[np.abs(G) <= 0.3] = 0
        G = (G != 0).astype("int")
        SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)  # SHD with thres as 0.3
        res[iter, 0] = SHD

        print(f"graph {iter + 1}: SHD(0.3): {SHD}, runtime: {res[iter, 1]: .4f}.")
        print("------------------------------ Done ------------------------------")
        np.savetxt(os.path.join(res_dir, "notears_res.txt"), res, delimiter=",")
    mean_res = np.mean(res, axis=0)
    std_res = np.std(res, axis=0)
    print(f"------------------------------ NOTEARS Results Summary: {graph_type}{s0} d{d} ------------------------------")
    print(f"MEAN SHD(0.3): {mean_res[0]: .1f} +/- {std_res[0]: .2f}\t MEAN TIME: {mean_res[1]: .1f} +/- {std_res[1]: .2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000,
                        help="sample size of the generated data")
    parser.add_argument('--graph_type', type=str, default='ER',
                        choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=2, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=10,
                        help="variable size to be generated")

    # parameter tuning
    parser.add_argument('--lamb1', type=float, default=0.0,
                        help="The weight of l1 norm regularization")
    parser.add_argument('--lamb2', type=float, default=0.1,
                        help="The weight of DAG uncertainty regularization")
    parser.add_argument('--cuda_device', type=str, default="0", choices=["0", "1", "2", "3", "4", "5", "6"])
    parser.add_argument('--thres_min', type=float, default=0.1, help="The minimum value of threshold.")
    parser.add_argument('--thres_max', type=float, default=0.75,help="The maximum value of threshold.")
    parser.add_argument('--data_type', type=str, default='homo', choices=['homo', 'hetero'], help='data type')
    args = parser.parse_args()

    # train_realdata(args)
    train_synthetic_notears(args)

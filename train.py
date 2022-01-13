import numpy as np
from linear_update_A_B import nll_linear_A_B as nll_linear_old
from linear_update_A_ab import nll_linear_A_a_b as nll_linear
import notears
from notears.linear import notears_linear
import time
import utils as ut
import copy
import argparse
import os


def train_synthetic_data(args):
    res = np.zeros([10, 6])

    # specify parameters
    n = args.sample_size
    thresholds = np.linspace(args.thres_min, args.thres_max, endpoint=True, num=1000)
    d = args.num_size
    graph_type = args.graph_type
    s0 = args.s0
    np.random.seed(123)
    data_type = args.data_type

    res_dir = f"results/{data_type}_poly/{graph_type}{s0}_d{d}_n{n}"
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    np.savetxt(os.path.join(res_dir, "nll_res.txt"), res, delimiter=",")

    for iter in range(10):
        print(f"--------------------------- graph {iter+1} ---------------------------")
        res = np.loadtxt(os.path.join(res_dir, "nll_res.txt"), delimiter=",")
        X = np.loadtxt(os.path.join(f"data/{data_type}_poly/{graph_type}{s0}_d{d}/data", f"X_{iter+1}.txt"), delimiter=",")
        X = X[:n, :]
        A_gt = np.loadtxt(os.path.join(f"data/{data_type}_poly/{graph_type}{s0}_d{d}/graph", f"A_{iter+1}.txt"), delimiter=",")

        np.random.seed(123)
        ut.set_random_seed(123)
        if args.formulation == 2:
            A_nll, a_nll, b_nll, t_nll, nll, rec = nll_linear(X=X,
                                                              A_gt=A_gt,
                                                              lamb1=0,
                                                              lamb2=0,
                                                              data_type=data_type,
                                                              verbose=True)
        elif args.formulation == 1:
            A_nll, B_nll, B0_nll, var_est, nll, rec, t_nll = nll_linear_old(X=X,
                                                                            A_gt=A_gt,
                                                                            lamb=args.lamb_a,
                                                                            verbose=True)
        res[iter, 1] = t_nll
        res[iter, 2] = nll
        res[iter, 3] = rec
        if args.formulation == 2:
            np.savetxt(os.path.join(res_dir, f"A_formulation{args.formulation}_nll_{iter + 1}.txt"), A_nll, delimiter=",")
            np.savetxt(os.path.join(res_dir, f"a_formulation{args.formulation}_nll_{iter + 1}.txt"), a_nll, delimiter=",")
            np.savetxt(os.path.join(res_dir, f"b_formulation{args.formulation}_nll_{iter + 1}.txt"), b_nll, delimiter=",")
        elif args.formulation == 1:
            np.savetxt(os.path.join(res_dir, f"A_formulation{args.formulation}_nll_{iter+1}.txt"), A_nll, delimiter=",")
            np.savetxt(os.path.join(res_dir, f"B_formulation{args.formulation}_nll_{iter + 1}.txt"), B_nll, delimiter=",")
            np.savetxt(os.path.join(res_dir, f"B0_formulation{args.formulation}_nll_{iter + 1}.txt"), B0_nll, delimiter=",")
            np.savetxt(os.path.join(res_dir, f"var_est_{iter + 1}.txt"), var_est, delimiter=",")


        G = copy.deepcopy(A_nll)
        G[np.abs(G) <= 0.3] = 0
        G = (G != 0).astype("int")
        SHD, _, _, _ = ut.count_accuracy(A_gt, G != 0)  # SHD with thres as 0.3
        res[iter, 0] = SHD

        print(f"graph {iter + 1}: SHD(0.3): {SHD}, nll loss: {nll: .4f}, rec loss: {rec: .4f}, runtime: {t_nll: .4f}.")
        print("------------------------------ Done ------------------------------")
        np.savetxt(os.path.join(res_dir, "nll_res.txt"), res, delimiter=",")
    mean_res = np.mean(res, axis=0)
    std_res = np.std(res, axis=0)
    print(f"------------------------------ Results Summary: {graph_type}{s0} d{d} ------------------------------")
    print(f"MEAN SHD(0.3): {mean_res[0]: .1f} +/- {std_res[0]: .2f}\t MEAN TIME: {mean_res[1]: .1f} +/- {std_res[1]: .2f}")


def train_real_data(args):
    # load the data
    # Sachs
    X = np.loadtxt("/Users/naiyuyin/Documents/Research/data/Sachs_data/sachs_X.txt", delimiter=",")
    W = np.loadtxt("/Users/naiyuyin/Documents/Research/data/Sachs_data/sachs_W.txt", delimiter=",")
    # SANGIOVESE
    # X = pd.read_table("//Users/naiyuyin/Documents/Research/data/SANGIOVESE/X.csv", sep = ",", index_col = 0).values
    # W = pd.read_table("//Users/naiyuyin/Documents/Research/data/SANGIOVESE/W.csv", sep = ",", index_col = 0).values
    # MAGIC-NIAB
    # X = pd.read_table("/Users/naiyuyin/Documents/Research/data/MAGIC-NIAB/X.csv", sep=",", index_col=0).values
    # W = pd.read_table("/Users/naiyuyin/Documents/Research/data/MAGIC-NIAB/W.csv", sep = ",", index_col = 0).values
    # ECOLI70
    # X = pd.read_table("/Users/naiyuyin/Documents/Research/data/ECOLI70/X.csv", sep=",", index_col=0).values
    # W = pd.read_table("/Users/naiyuyin/Documents/Research/data/ECOLI70/W.csv", sep = ",", index_col = 0).values

    # print("Apply notears linear formulation.")
    # start_time = time.time()
    # A_notears = notears_linear(X=X, lambda1=0, loss_type="l2")
    # # print(A_notears)
    # end_time = time.time()
    # G = copy.deepcopy(A_notears)
    # G[np.abs(G) <= 0.3] = 0
    # G = (G != 0).astype("int")
    # SHD, _, _, _ = ut.count_accuracy(W, G != 0)
    # print(f"NOTEARS results: SHD: {SHD}, Runtime: {end_time - start_time: .4f}.")
    # np.savetxt("results/A_notears_real.txt", A_notears, delimiter=",")

    print("Apply our formulation.")
    A_nll, a_nll, b_nll, t_nll, nll, rec = nll_linear(X=X,
                                                      A_gt=W,
                                                      lamb1=args.lamb_a,
                                                      lamb2=args.lamb_b,
                                                      data_type=args.data_type,
                                                      verbose=True)
    G = copy.deepcopy(A_nll)
    G[np.abs(G) <= 0.3] = 0
    G = (G != 0).astype("int")
    SHD, _, _, _ = ut.count_accuracy(W, G != 0)
    print(f"Our results: SHD: {SHD}, Runtime: {t_nll: .4f}.")
    np.savetxt("results/A_nll.txt", A_nll, delimiter=",")
    # np.savetxt("var_est.txt", var_est, delimiter=",")
    # np.savetxt("loss.txt", losses, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000,
                        help="sample size of the generated data")
    parser.add_argument('--graph_type', type=str, default='ER',
                        choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=2, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=10,
                        help="variable size to be generated")
    parser.add_argument('--data_type', type=str, default='hetero', choices=['homo', 'hetero'], help='data type')
    parser.add_argument('--formulation', type=int, default=1, choices=[1, 2], help='formuation')

    # parameter tuning
    parser.add_argument('--lamb_a', type=float, default=0.0,
                        help="The coefficient of l1 regularization on variance distribution lower bounds a.")
    parser.add_argument('--lamb_b', type=float, default=0.0,
                        help="The coefficient of l1 regularization on variance distribution interval b.")
    parser.add_argument('--cuda_device', type=str, default="0", choices=["0", "1", "2", "3", "4", "5", "6"])

    # evaluation parameters
    parser.add_argument('--thres_min', type=float, default=0.1, help="The minimum value of threshold.")
    parser.add_argument('--thres_max', type=float, default=0.75, help="The maximum value of threshold.")

    args = parser.parse_args()

    # train_real_data(args)
    train_synthetic_data(args)
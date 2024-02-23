import numpy as np
import argparse
import utils as ut
import os


def hetero_linear_synthetic_data_generation(d, n, e, graph_type, A_range, B_range, B0_range):
    G = ut.simulate_dag(d, e, graph_type)
    A = ut.simulate_parameter(G, w_ranges=((-A_range, -0.5), (0.5, A_range)))
    A_0 = np.random.uniform(-A_range, A_range, size=(d,))
    B_0 = np.random.uniform(-B0_range, B0_range, size=(d,))
    B = ut.simulate_parameter(G, w_ranges=((-B_range, -0.5), (0.5, B_range)))
    X, Var = ut.simulate_linear_neq(A, A_0, B, B_0, n)
    return X, A, A_0, B, B_0, G, Var 


def homo_linear_synthetic_data_generation(d, n, e, graph_type, a_range):
    G = ut.simulate_dag(d, e, graph_type)
    A = ut.simulate_parameter(G, w_ranges=((-a_range, -0.5), (0.5, a_range)))
    # A_0 = np.random.uniform(-a_range, a_range, size=(d,))
    X = ut.simulate_linear_sem(A, n, 'gauss', noise_scale=None)
    return X, A, G


def main_linear_syn(args):
    # specify parameters
    d = args.num_size
    n = args.sample_size
    s0 = args.s0
    graph_type = args.graph_type
    data_type = args.data_type
    ut.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if data_type == "hetero_linear":
        A_range = 2.0
        B_range = 1.0
        B0_range = 0.5
    else:
        A_range = 2.0

    # generate folder for the generated data
    print(f"Generate {data_type} data with {d} nodes, {n} samples from {graph_type}{s0} graph.")
    path = f"data/{data_type}"
    file_dir = f"data/{data_type}/{graph_type}{s0}_d{d}"
    data_dir = f"data/{data_type}/{graph_type}{s0}_d{d}/data"
    graph_dir = f"data/{data_type}/{graph_type}{s0}_d{d}/graph"
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
        os.mkdir(data_dir)
        os.mkdir(graph_dir)

    for t in range(10):
        if data_type == "hetero_old":
            X, A, A_0, B, B_0, G, Var = hetero_old_synthetic_data_generation(d, n, s0 * d, graph_type, A_range, B_range,B0_range)
            np.savetxt(os.path.join(data_dir, f"X_{t + 1}.txt"), X, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"A_{t + 1}.txt"), A, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"A0_{t + 1}.txt"), A_0, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"B_{t + 1}.txt"), B, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"B0_{t + 1}.txt"), B_0, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"G_{t + 1}.txt"), G, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"Var_{t + 1}.txt"), Var, delimiter=",")

        elif data_type == "hetero_new":
            X, A, A0, a, b, G, Var = hetero_new_synthetic_data_generation(d, n, s0 * d, graph_type, A_range, a_range, b_range)
            np.savetxt(os.path.join(data_dir, f"X_{t+1}.txt"), X, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"A_{t+1}.txt"), A, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"A0_{t + 1}.txt"), A0, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"lb_{t + 1}.txt"), a, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"ub_{t + 1}.txt"), b, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"G_{t + 1}.txt"), G, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"Var_{t + 1}.txt"), Var, delimiter=",")

        elif data_type == "homo":
            X, A, G = homo_synthetic_data_generation(d, n, s0 * d, graph_type, A_range)
            np.savetxt(os.path.join(data_dir, f"X_{t + 1}.txt"), X, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"A_{t + 1}.txt"), A, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"G_{t + 1}.txt"), G, delimiter=",")

        elif data_type == "hetero_new2":
            X, A, A0, a, B, Var = hetero_new2_synthetic_data_generation(d, n, s0*d, graph_type, A_range)
            np.savetxt(os.path.join(data_dir, f"X_{t+1}.txt"), X, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"A_{t + 1}.txt"), A, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"A0_{t + 1}.txt"), A0, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"a_{t + 1}.txt"), np.array([a]), delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"B_{t + 1}.txt"), B, delimiter=",")
            np.savetxt(os.path.join(graph_dir, f"Var_{t + 1}.txt"), Var, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')
    # parameters to generate graphs
    parser.add_argument('--graph_type', type=str, default='ER',
                        choices=['ER', 'SF'])
    parser.add_argument('--s0', type=int, default=3, help="degree of variables")
    parser.add_argument('--num_size', type=int, default=10,
                        help="variable size to be generated")
    # parameters to generate data
    parser.add_argument('--data_type', type=str, default='hetero_old', choices=['homo', 'hetero_old', 'hetero', 'hetero_new2'],
                        help='data type')
    parser.add_argument('--sample_size', type=int, default=5000,
                        help="sample size of the generated data")
    args = parser.parse_args()
    main(args)

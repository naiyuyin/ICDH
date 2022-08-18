import numpy as np
import utils as ut
import copy
import os
import pdb

graph = "SF"
s0 = 4
data = "hetero"
d = 50

res = np.zeros([10, 1])
ind = [0,1,2,3,4,5,6,7,8,9]
for it in ind:
    A_gt = np.loadtxt("data/" + data + "/" + graph + str(s0) + "_d" + str(d) + "/graph/A_" + str(it + 1) + ".txt", delimiter=",")
    path = "../GraN-DAG/hetero/grandag++/" + graph + str(s0) + "_d" + str(d) + "_n1000/res"+str(it+1)+"/cam-pruning/"
    all_files = [x[0] for x in os.walk(path) if x[0] is not path]
    shd_best = np.inf
    for f in all_files:
    	A_est = np.load(f +'/DAG.npy')
    	shd, _, _, _ = ut.count_accuracy(A_gt, A_est)
    	if shd < shd_best:
    		shd_best = shd
    		del shd
    res[it,0] = shd_best
    
    # A_est = np.load("/Users/naiyuyin/Documents/Research/IBM/GraN-DAG/hetero/grandag++/" + graph + str(s0) + "_d" + str(d) + "_n1000/res"+str(iter+1)+"/to-dag/DAG.npy")
    # res[iter, 0], _, _, _ = ut.count_accuracy(A_gt, A_est != 0)
    # G = copy.deepcopy(A_est)
    # G[np.abs(G) <= 0.3] = 0
    # # G, _ = ut.threshold_till_dag(G)
    # G = (G != 0).astype("int")
    # res[iter, 0], _, _, _ = ut.count_accuracy(A_gt, G != 0)
print(res)
print(np.mean(res[ind, :], axis = 0 ))
print(np.std(res[ind,:], axis = 0))
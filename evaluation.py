import numpy as np
import utils as ut
import copy

graph = "SF"
s0 = 4
data = "homo_nv"
d =50
thres = np.linspace(0.20, 0.75, endpoint=True, num=1000)
print("ours")
res_nll = np.zeros([10, 4])
# ind = [0,1,2,3,4,5,6,7,8,9]
for it in range(10):
    # print(it+1)
    A_gt = np.loadtxt("data/" + data + "/" + graph + str(s0) + "_d" + str(d) + "/W_" + str(it + 1) + ".txt", delimiter=",")
    A_nll = np.loadtxt("results/"+ data + "/" + graph + str(s0) + "_d" + str(d) + "_n1000/A_formulation1_nll_" + str(it + 1) + ".txt", delimiter=",")

    _, _, aucPR_nll = ut.PrecisionRecall_curve(A_gt, A_nll)
    res_nll[it, 3] = aucPR_nll

    try:
        nll_shds, auc_nll = ut.shd_curve(A_gt, A_nll, thres, post=False)
        res_nll[it, 0] = auc_nll
        res_nll[it, 2] = np.min(nll_shds)

    except ValueError:
        print("Encounter error.")
        pass

    G = copy.deepcopy(A_nll)
    G[np.abs(G) <= 0.3] = 0
    G, _ = ut.threshold_till_dag(G)
    G = (G != 0).astype("int")
    res_nll[it, 1], _, _, _ = ut.count_accuracy(A_gt, G != 0)  # SHD with thres as 0.3


mean_res_nll= np.mean(res_nll, axis= 0)
std_res_nll = np.std(res_nll, axis= 0)
# print(mean_res_nll)
# print(std_res_nll)

# res_notears = np.zeros([10, 4])
# print("notears")
# for iter in range(10):
#     # print(iter + 1)
#     A_gt = np.loadtxt("data/" + data + "/" + graph + str(s0) + "_d" + str(d) + "/W_" + str(iter + 1) + ".txt", delimiter=",")
#     A_notears = np.loadtxt("results/"+ data + "/" + graph + str(s0) + "_d" + str(d) + "_n1000/A_notears_" + str(iter + 1) + ".txt", delimiter=",")

#     _, _, aucPR_notears= ut.PrecisionRecall_curve(A_gt, A_notears)
#     res_notears[iter, 3] = aucPR_notears

#     try:
#         notears_shds, auc_notears = ut.shd_curve(A_gt, A_notears, thres, post=False)
#         res_notears[iter, 0] = auc_notears
#         res_notears[iter, 2] = np.min(notears_shds)
#     except ValueError:
#         print("Encounter error.")
#         pass
#     G = copy.deepcopy(A_notears)
#     G[np.abs(G) <= 0.3] = 0
#     G,_ = ut.threshold_till_dag(G)
#     G = (G != 0).astype("int")
#     res_notears[iter, 1], _, _, _ = ut.count_accuracy(A_gt, G != 0)  # SHD with thres as 0.3

    
# mean_res_notears= np.mean(res_notears, axis= 0)
# std_res_notears = np.std(res_notears, axis= 0)
# print(mean_res_notears)
# print(std_res_notears)


# res = np.zeros([10, 4])
# print("golem-Nv")
# for iter in range(10):
#     print(iter + 1)
#     A_gt = np.loadtxt("data/" + data + "/" + graph + str(s0) + "_d" + str(d) + "/W_" + str(iter + 1) + ".txt", delimiter=",")
#     A_est = np.loadtxt("/gpfs/u/home/CDPL/CDPLynny/scratch-shared/golem/output/homo_nv/hetero/" + graph + str(s0) + "_d" + str(d) + "/A_golem_est_" + str(iter + 1) + ".txt", delimiter=",")
#     A_p = np.loadtxt("/gpfs/u/home/CDPL/CDPLynny/scratch-shared/golem/output/homo_nv/hetero/" + graph + str(s0) + "_d" + str(d) + "/A_golem_processed_" + str(iter + 1) + ".txt", delimiter=",")

    # A_est = np.loadtxt("/Users/naiyuyin/Documents/Research/IBM/golem/output/hetero/"+graph+str(s0)+"_d"+str(d)+"/A_golem_est_"+str(iter+1)+".txt", delimiter=",")
    # A_p = np.loadtxt("/Users/naiyuyin/Documents/Research/IBM/golem/output/hetero/"+graph+str(s0)+"_d"+str(d)+"/A_golem_processed_"+str(iter+1)+".txt", delimiter=",")

    # pre, rec, aucPR = ut.PrecisionRecall_curve(A_gt, A_est)
    # res[iter, 3] = aucPR

    # try:
    #     golem_shds, auc = ut.shd_curve(A_gt, A_est, thres, post=True)
    #     res[iter, 0] = auc
    #     res[iter, 2] = np.min(golem_shds)
    # except ValueError:
    #     print("Encounter error.")
    #     pass

    # G = copy.deepcopy(A_est)
    # G[np.abs(G) <= 0.3] = 0
    # # G, _ = ut.threshold_till_dag(G)
    # G = (G != 0).astype("int")
    # res[iter, 1], _, _, _ = ut.count_accuracy(A_gt, G != 0)

    
# print(res)
# mean_res_golem = np.mean(res, axis = 0)
# std_res_golem = np.std(res, axis = 0)
# print(mean_res_golem)

# res = np.zeros([10, 1])
# ind = [1,2,3,4,5,6,8,9]
# for iter in ind:
#     A_gt = np.loadtxt("data/" + data + "/" + graph + str(s0) + "_d" + str(d) + "/graph/A_" + str(iter + 1) + ".txt", delimiter=",")
#     A_est = np.load("/Users/naiyuyin/Documents/Research/IBM/GraN-DAG/hetero/grandag++/" + graph + str(s0) + "_d" + str(d) + "_n1000/res"+str(iter+1)+"/to-dag/DAG.npy")
#     res[iter, 0], _, _, _ = ut.count_accuracy(A_gt, A_est != 0)

# print(np.mean(res[ind, :], axis = 0 ))
print("Ours: %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f." % (mean_res_nll[0], std_res_nll[0], mean_res_nll[1], std_res_nll[1],mean_res_nll[3], std_res_nll[3]))
# print("NOTEARS: %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f." % (mean_res_notears[0], std_res_notears[0], mean_res_notears[1], std_res_notears[1],mean_res_notears[3], std_res_notears[3]))
# print("golem-nv: %.2f +/- %.2f, %.2f +/- %.2f, %.2f +/- %.2f." % (mean_res_golem[0], std_res_golem[0], mean_res_golem[1], std_res_golem[1],mean_res_golem[3], std_res_golem[3]))

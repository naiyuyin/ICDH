import numpy as np
import os 
from linear_update_A_B import nll_linear_A_B as nll_linear
from linear_update_A_ab import nll_linear_A_a_b as nll_linear_new
from notears.linear import notears_linear

# X = np.loadtxt(f"data/real/pairs/pair0001.txt", delimiter = " ")
# print(X.shape)
# A_notears = notears_linear(X=X, lambda1=0, loss_type="l2")
# print(A_notears)
# A_nll, B_nll, B0_nll, var_est, nll, rec, t_nll = nll_linear(X=X, A_gt=np.array([[0,1],[0,0]]), lamb=0,  verbose=True)
# print(A_nll)

bi_sets = np.loadtxt(f"data/real/pairs/processed/bi_sets.txt", delimiter = ",")
ce = np.loadtxt(f"data/real/pairs/processed/ce.txt", delimiter = ",")
label = ce[:,0]
weights = ce[:,1]
pred = np.zeros([len(bi_sets),3])
c = 0
acc_notears = 0 
acc_nll = 0 
total = 0
num_notears = 0
num_nll = 0

for s in bi_sets:
	s = int(s)
	pred[c,2] = label[s-1]
	total += weights[s-1]
	print(f"######################################## dataset {s:04d} ########################################")
	X = np.loadtxt(f"data/real/pairs/processed/X_{s:04d}.txt", delimiter = ",")
	A_notears = notears_linear(X=X, lambda1=0, loss_type="l2")
	if np.abs(A_notears[0,1]) > np.abs(A_notears[1,0]):
		pred[c, 0] = 1 
		print(f"Notears prediction: X --> Y")
	elif np.abs(A_notears[0,1]) < np.abs(A_notears[1,0]):
		pred[c, 0] = 0 
		print(f"Notears prediction: X <-- Y")
	if label[s-1] == pred[c,0]:
		print(f"Notears predicts CORRECT!!!")
		acc_notears += weights[s-1]
		num_notears += 1
	else:
		print(f"Notears predicts WRONG")
	if label[s-1] == 1:
		A_gt = np.array([[0,1],[0,0]])
	elif label[s-1] == 0:
		A_gt = np.array([[0,0],[1,0]])
	# A_nll, B_nll, B0_nll, var_est, nll, rec, t_nll = nll_linear(X=X, A_gt = A_gt, lamb=0, verbose=False)
	A_nll, a_nll, b_nll, t_nll, nll, rec = nll_linear_new(X=X, A_gt=A_gt, data_type="hetero", verbose=False)
	if np.abs(A_nll[0,1]) > np.abs(A_nll[1,0]):
		pred[c, 1] = 1 
		print(f"Our method prediction: X --> Y")
	elif np.abs(A_nll[0,1]) < np.abs(A_nll[1,0]):
		pred[c, 1] = 0 
		print(f"Our method prediction: X <-- Y")
	if label[s-1] == pred[c,1]:
		print(f"Our method predicts CORRECT!!!")
		acc_nll += weights[s-1]
		num_nll += 1
	else:
		print(f"Our method predicts WRONG")
	c += 1
	print(f"#################################################################################################\n\n\n")


print(f"Notears correctly predict {num_notears}/99 cause-effect relations, accuracy: {acc_notears / total: .4f}.")
print(f"Our method correctly predict {num_nll}/99 cause-effect relations, accuracy: {acc_nll / total: .4f}.")
# print(f"Notears accuracy: {acc_notears / total: .4f}.")
# print(f"Our method accuracy: {acc_nll / total: .4f}.")
if not os.path.isdir(f"results/real"):
	os.mkdir(f"results/real")
np.savetxt(f"results/real/pairs_pred.txt", pred, delimiter = ",")
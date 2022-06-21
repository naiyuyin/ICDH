import numpy as np
from linear_update_A_B import nll_linear_A_B as nll_linear
from notears.linear import notears_linear

# X = np.loadtxt(f"data/real/pairs/pair0001.txt", delimiter = " ")
# print(X.shape)
# A_notears = notears_linear(X=X, lambda1=0, loss_type="l2")
# print(A_notears)
# A_nll, B_nll, B0_nll, var_est, nll, rec, t_nll = nll_linear(X=X, A_gt=np.array([[0,1],[0,0]]), lamb=0,  verbose=True)
# print(A_nll)

bi_sets = np.loadtxt(f"/Users/naiyuyin/Box/Naiyu Yin/nllSEM/data/real/pairs/processed/bi_sets.txt", delimiter = ",")
label = np.loadtxt(f"/Users/naiyuyin/Box/Naiyu Yin/nllSEM/data/real/pairs/processed/ce.txt", delimiter = ",")
pred = np.zeros([len(bi_sets),2])
c = 0
acc_notears = 0 
acc_nll = 0 

for s in bi_sets:
	print(f"######################################## dataset {s:04d} ########################################")
	X = np.loadtxt(f"data/real/processed/X_{s:04d}.txt", delimiter = ",")
	A_notears = notears_linear(X=X, lambda1=0, loss_type="l2")
	if A_notears[0,1] > A_notears[1,0]:
		pred[c, 0] = 1 
		print(f"Notears prediction: X --> Y")
	elif A_notears[0,1] > A_notears[1,0]:
		pred[c, 0] = 0 
		print(f"Notears prediction: X <-- Y")
	if label[c] == pred[c,0]:
		print(f"Notears predicts CORRECT!!!")
	else:
		print(f"Notears predicts WRONG")
	if label[c] == 1:
		A_gt = np.array([[0,1],[0,0]])
	elif label[c] == 0:
		A_gt = np.array([[0,0],[1,0]])
	A_nll, B_nll, B0_nll, var_est, nll, rec, t_nll = nll_linear(X=X, A_gt = A_gt, lamb=0, verbose=False)
	if A_nll[0,1] > A_nll[1,0]:
		pred[c, 1] = 1 
		print(f"Our method prediction: X --> Y")
	elif A_nll[0,1] > A_nll[1,0]:
		pred[c, 1] = 0 
		print(f"Our method prediction: X <-- Y")
	if label[c] == pred[c,1]:
		print(f"Our method predicts CORRECT!!!")
	else:
		print(f"Our method predicts WRONG")
	c += 1
	print(f"#################################################################################################\n\n\n")


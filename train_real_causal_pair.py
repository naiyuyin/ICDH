import numpy as np
from linear_update_A_B import nll_linear_A_B as nll_linear
from notears.linear import notears_linear
from notears.nonlinear import notears_nonlinear
from notears.nonlinear import NotearsMLP
from Nonlinear_update import MLP
from Nonlinear_update import Nonlinear_update
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# X = np.loadtxt(f"data/real/pairs/pair0001.txt", delimiter = " ")
# print(X.shape)
# A_notears = notears_linear(X=X, lambda1=0, loss_type="l2")
# print(A_notears)
# A_nll, B_nll, B0_nll, var_est, nll, rec, t_nll = nll_linear(X=X, A_gt=np.array([[0,1],[0,0]]), lamb=0,  verbose=True)
# print(A_nll)

bi_sets = np.loadtxt(f"data/real/bi_sets.txt", delimiter=",")
label = np.loadtxt(f"data/real/ce.txt", delimiter=",")
meta = np.loadtxt(f"data/real/pairmeta.txt")
pred = np.zeros([len(bi_sets),2])
c = 0
acc_notears = 0
correct_notears = 0
acc_nll = 0
correct_our = 0

for s in bi_sets:
	if s >= 103:
		print(f"######################################## dataset {int(s):04d} ########################################")
		X = np.loadtxt(f"data/real/X_{int(s):04d}.txt", delimiter = ",")
		# A_notears = notears_linear(X=X, lambda1=0, loss_type="l2")
		torch.set_default_dtype(torch.double)
		np.set_printoptions(precision=3)
		model = NotearsMLP(dims=[2, 5, 1], bias=True)
		A_notears = notears_nonlinear(model, X, lambda1=0, lambda2=0)
		if A_notears[0,1] > A_notears[1,0]:
			pred[c, 0] = 1
			print(f"Notears prediction: X --> Y")
		elif A_notears[0,1] < A_notears[1,0]:
			pred[c, 0] = 0
			print(f"Notears prediction: X <-- Y")
		if label[c] == pred[c,0]:
			print(f"Notears predicts CORRECT!!!")
			acc_notears += meta[meta[:,0] == s, -1]
			correct_notears += 1
		else:
			print(f"Notears predicts WRONG")
		if label[c] == 1:
			A_gt = np.array([[0,1],[0,0]])
		elif label[c] == 0:
			A_gt = np.array([[0,0],[1,0]])
		# A_nll, B_nll, B0_nll, var_est, nll, rec, t_nll = nll_linear(X=X, A_gt = A_gt, lamb=0, verbose=False)
		model = MLP(dims=[2, 5, 1], bias=False)
		A_nll, nlls = Nonlinear_update(model=model, X=X, lamb1=0.05, lamb2=0.05, W_true=A_gt, verbose=False)
		if A_nll[0,1] > A_nll[1,0]:
			pred[c, 1] = 1
			print(f"Our method prediction: X --> Y")
		elif A_nll[0,1] < A_nll[1,0]:
			pred[c, 1] = 0
			print(f"Our method prediction: X <-- Y")
		if label[c] == pred[c, 1]:
			print(f"Our method predicts CORRECT!!!")
			acc_nll += meta[meta[:, 0] == s, -1]
			correct_our += 1
		else:
			print(f"Our method predicts WRONG")
		c += 1

		print(f"notears: correct {correct_notears}/99, weights {acc_notears}\nours: correct {correct_our}/99, weights {acc_nll}.")
		print(f"#################################################################################################\n\n\n")


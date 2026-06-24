import h5py
import numpy as np
import pickle
from scipy.integrate import trapezoid as trapz
from sklearn.metrics import accuracy_score, roc_curve
import sys

from common.preset import *
from model.MLP import MLP_Baseline


def get_perm_data(inp):
	N = len(inp)
	X = np.zeros((N * MLP_BASELINE_NPERM, inp.shape[1] * inp.shape[2]))
	for i in range(N):
		for j in range(MLP_BASELINE_NPERM):
			raw_inp = inp[i]
			perm = np.random.permutation(len(raw_inp))
			X[i * MLP_BASELINE_NPERM + j, :] = raw_inp[perm, :].flatten()
	return X


if __name__ == "__main__":
	op_index = int(sys.argv[1])
	f = h5py.File(DATA_PATH, 'r')
	inp_train = f['inp_train'][:]
	Y_train = f['Y_train'][:]
	inp_test = f['inp_test'][:]
	Y_test = f['Y_test'][:]

	# training
	X_train = get_perm_data(inp_train)
	Y_train = np.repeat(Y_train, MLP_BASELINE_NPERM, axis=0)
	clf = MLP_Baseline()
	clf.fit(X_train, Y_train[:, op_index])

	# testing
	X_test = get_perm_data(inp_test)
	prediction  = clf.predict_proba(X_test)[:,1]
	prediction = prediction.reshape((len(inp_test), MLP_BASELINE_NPERM))
	prediction = np.mean(prediction, axis=1)
	target = Y_test[:, op_index]
	accuracy = accuracy_score(prediction > 0.5, target)
	fpr, tpr, thresholds = roc_curve(target, prediction)
	auc = trapz(tpr, x=fpr)

	print('accuracy:', accuracy)
	print('AUC:', auc)

	pickle.dump((clf, accuracy, auc, [fpr, tpr, thresholds]), open(MLP_BASELINE_PATH_TEMPLATE.format(NVAR, op_index), 'wb'))
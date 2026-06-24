import h5py
import numpy as np
import pickle

from common.preset import *
from common.symbolic_tree import OPERATOR_LIST
from model.GA import GA_Baseline

def run_genetic_algorithm(X, Y):
	sr = GA_Baseline()
	sr.fit(X, Y)
	prediction = np.array(
		[sr._program.__str__().find(op) >= 0 for op in OPERATOR_LIST]
	)
	return prediction


if __name__ == "__main__":
	f = h5py.File(DATA_PATH, 'r')
	inp = f['inp_test'][:]
	Y = f['Y_test'][:]

	result = np.zeros_like(Y)
	for i in range(len(inp)):
		result[i] = run_genetic_algorithm(inp[i, :, :-1], inp[i, :, -1])
		print('\r', i+1, '/', len(inp), end='')
	print('')

	acc = np.mean(Y == result, axis=0)
	pickle.dump(acc, open(GA_BASELINE_PATH_TEMPLATE.format(NVAR), 'wb'))

	print('accuracies:')
	for i in range(len(OPERATOR_LIST)):
		print(OPERATOR_LIST[i], ':', acc[i])

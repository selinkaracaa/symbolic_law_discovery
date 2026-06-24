import argparse
import h5py
import numpy as np
import pickle

from common.symbolic_tree import OPERATOR_LIST
from common.preset import *
from model.GA import GA


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run genetic algorithm for the specified formula')
	parser.add_argument('--formula', help='name of formula used')
	parser.add_argument('--path', default='default', help='path that the experiment data is saved')
	parser.add_argument('--result_path', default='default', help='path that the genetic algorithm result is saved')
	args = parser.parse_args()

	if args.path == 'default':
		args.path = EXPERIMENT_PATH_TEMPLATE.format(args.formula)
	if args.result_path == 'default':
		args.result_path = EXPERIMENT_RESULT_PATH_TEMPLATE.format(args.formula)

	f = h5py.File(args.path, 'r')
	X = f['inp'][:]
	pred = f['prediction'][:]


	program = []
	program_baseline = []
	score = []
	score_baseline = []

	for i in range(EXPERIMENT_NTRIALS):
		func_set_baseline = tuple(OPERATOR_LIST)
		func_set = tuple(np.array(OPERATOR_LIST)[np.random.uniform(0,1, size=6) < pred])
		if len(func_set) == 0:
			func_set = func_set_baseline
		regressor = GA(func_set, EXPERIMENT_NGEN)
		regressor_baseline = GA(func_set_baseline, EXPERIMENT_NGEN)
		regressor.fit(X[: :-1], X[:, -1])
		regressor_baseline.fit(X[:, :-1], X[:, -1])

		program.append(regressor._program.__str__())
		program_baseline.append(regressor_baseline._program.__str__())
		score.append(regressor.score(X[i, :, :-1], X[i, :, -1]))
		score_baseline.append(regressor_baseline.score(X[i, :, :-1], X[i, :, -1]))

	pickle.dump((program, program_baseline, score, score_baseline), open(args.result_path, 'wb'))
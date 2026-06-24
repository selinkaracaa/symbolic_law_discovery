import argparse
import numpy as np
import h5py
import pickle
from common.symbolic_tree import OPERATOR_LIST
from common.preset import *
from model.GA import GA


parser = argparse.ArgumentParser(description='Run genetic algorithm on the dataset')

parser.add_argument('--path', default=DATA_PATH, 
	help='data path, containing model predictions')
parser.add_argument('--result_path', default=GA_RESULT_PATH, 
	help='result path for the genetic algorithm')


args = parser.parse_args()

f = h5py.File(args.path, 'r')

generations = [4, 8, 12, 16, 20]

X_test = f['inp_test'][:]
pred_test = f['prediction_test'][:]
ground_truth = f['funcs_test'][:]

all_program = []
all_program_baseline = []
all_score = []
all_score_baseline = []


for g in generations:
	program = []
	program_baseline = []
	score = []
	score_baseline = []
	for i in range(len(X_test)):
		func_set_baseline = tuple(OPERATOR_LIST)
		func_set = tuple(np.array(OPERATOR_LIST)[pred_test[i] > 0.5])
		if len(func_set) == 0:
			func_set = func_set_baseline
		regressor = GA(func_set, g)
		regressor_baseline = GA(func_set_baseline, g)
		regressor.fit(X_test[i, :, :-1], X_test[i, :, -1])
		regressor_baseline.fit(X_test[i, :, :-1], X_test[i, :, -1])

		program.append(regressor._program.__str__())
		program_baseline.append(regressor_baseline._program.__str__())
		score.append(regressor.score(X_test[i, :, :-1], X_test[i, :, -1]))
		score_baseline.append(regressor_baseline.score(X_test[i, :, :-1], X_test[i, :, -1]))

	all_program.append(program)
	all_program_baseline.append(program_baseline)
	all_score.append(score)
	all_score_baseline.append(score_baseline)

f.close()
print("GA results saved to:", args.result_path)
pickle.dump((all_program, all_program_baseline, all_score, all_score_baseline, ground_truth), open(args.result_path, 'wb'))
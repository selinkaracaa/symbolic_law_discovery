import argparse
import matplotlib.pyplot as plt
import h5py
from itertools import combinations

from common.preset import *
from common.symbolic_tree import OPERATOR_LIST
from experiment.formulas import get_formula_variables


def get_plot_arrangement(k):
	if k == 2:
		return 1, 1
	elif k == 3:
		return 1, 3
	elif k == 5:
		return 2, 5
	elif k == 10:
		return 5, 9


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Plot the encodings for the specified formula')
	parser.add_argument('--formula', help='name of formula used')
	parser.add_argument('--path', default='default', help='path that the experiment data is saved')
	args = parser.parse_args()
	if args.path == 'default':
		args.path = EXPERIMENT_PATH_TEMPLATE.format(args.formula)

	variables = get_formula_variables(args.formula)

	f = h5py.File(args.path, 'r')
	X = f['encoding'][:]
	projected = len(X.shape) == 3

	if projected:
		plt.rc('font', size=10)
		att = f['attention'][:]
		nrow, ncol = get_plot_arrangement(len(variables))
		fig, axes = plt.subplots(2 * nrow, ncol, gridspec_kw={'height_ratios': [2, 0.8] * nrow}, figsize=(4 * ncol, 6 * nrow))

		for i, (ax1, ax2) in enumerate(np.hstack([axes[2*i:2*i+2, :] for i in range(nrow)]).T):
			var_pair = [
				variables[list(combinations(range(len(variables)), 2))[i][0]],
				variables[list(combinations(range(len(variables)), 2))[i][1]],
			]
			ax1.imshow(X[i], origin='lower')

			ax1.set_xticks(
				np.arange(0, HIGH_RESOLUTION[1] + HIGH_RESOLUTION[1] // 4, HIGH_RESOLUTION[1] // 4) - 0.5,
			)
			ax1.set_xticklabels(
				np.arange(
					VAR_RANGE_LOW, 
					VAR_RANGE_HIGH + (VAR_RANGE_HIGH - VAR_RANGE_LOW) //4, 
					(VAR_RANGE_HIGH - VAR_RANGE_LOW) //4,
				)
			)
			ax1.set_yticks(
				np.arange(0, HIGH_RESOLUTION[0] + HIGH_RESOLUTION[0] // 4, HIGH_RESOLUTION[0] // 4) - 0.5, 
			)
			ax1.set_yticklabels(
				np.arange(
					VAR_RANGE_LOW, 
					VAR_RANGE_HIGH + (VAR_RANGE_HIGH - VAR_RANGE_LOW) //4, 
					(VAR_RANGE_HIGH - VAR_RANGE_LOW) //4,
				)
			)
			ax2.bar(range(len(OPERATOR_LIST)), att[:, i])
			ax2.set_xticks(range(len(OPERATOR_LIST)))
			ax2.set_xticklabels(OPERATOR_LIST)
			ax2.set_ylim(0, 1)

			ax1.set_xlabel(var_pair[1])
			ax1.set_ylabel(var_pair[0], labelpad=-8)

		for i in range(nrow):
			axes[2 * i + 1, 0].set_ylabel('attention weight')

		plt.show()

	else:
		plt.rc('font', size=15)
		plt.rc('figure', figsize=(10, 10))  
		plt.imshow(X, origin='lower')
		plt.xticks(
			np.arange(0, HIGH_RESOLUTION[1] + HIGH_RESOLUTION[1] // 4, HIGH_RESOLUTION[1] // 4) - 0.5, 
			np.arange(
				VAR_RANGE_LOW, 
				VAR_RANGE_HIGH + (VAR_RANGE_HIGH - VAR_RANGE_LOW) //4, 
				(VAR_RANGE_HIGH - VAR_RANGE_LOW) //4,
			)
		)

		plt.yticks(
			np.arange(0, HIGH_RESOLUTION[0] + HIGH_RESOLUTION[0] // 4, HIGH_RESOLUTION[0] // 4) - 0.5, 
			np.arange(
				VAR_RANGE_LOW, 
				VAR_RANGE_HIGH + (VAR_RANGE_HIGH - VAR_RANGE_LOW) //4, 
				(VAR_RANGE_HIGH - VAR_RANGE_LOW) //4,
			)
		)
		plt.xlabel(variables[1])
		plt.ylabel(variables[0])
		plt.show()
import h5py
import numpy as np
from itertools import combinations

from common.data_split import data_split
from common.preset import *
from common.symbolic_tree import *
from common.write_h5_file import write_h5_file


def add_noise(x, noise=NOISE):
	return x + np.random.normal(0, noise, size=x.shape)


def generate_data_from_tree(
		root, 
		nvar=NVAR,
		nsample=NSAMPLE,
		var_range_low=VAR_RANGE_LOW, 
		var_range_high=VAR_RANGE_HIGH):

	X = np.random.uniform(
		low=var_range_low, 
		high=var_range_high, 
		size=(nsample, nvar)
	)
	Y = np.array([get_value(root, x) for x in X])[:, None]

	return X, Y


def generate_data_instance(
		nvar=NVAR,
		nsample=NSAMPLE,
		var_range_low=VAR_RANGE_LOW, 
		var_range_high=VAR_RANGE_HIGH, 
		terminal_prob=TERMINAL_PROB,
		constant_prob=CONSTANT_PROB,
		constant_range=CONSTANT_RANGE,
		min_depth=MIN_DEPTH,
		max_depth=MAX_DEPTH,
		y_thresh=Y_THRESH):

	while True:
		root = generate_random_tree(
			nvar, terminal_prob, constant_prob, constant_range,
			[min_depth, max_depth],
		)

		X, Y = generate_data_from_tree(
			root, nvar=nvar, nsample=nsample,
			var_range_low=var_range_low, 
			var_range_high=var_range_high,
		)

		if np.abs(Y).max() > y_thresh:
			continue

		Y = add_noise(Y)
		inp = np.append(X, Y, axis=1)
		oup = get_classification_labels(root)
		f = get_symbolic_form(root)

		return inp, oup, f


def generate_data(
		ntree=NTREE,
		nvar=NVAR,
		nsample=NSAMPLE,
		operator_list=OPERATOR_LIST,
		var_range_low=VAR_RANGE_LOW, 
		var_range_high=VAR_RANGE_HIGH, 
		terminal_prob=TERMINAL_PROB,
		constant_prob=CONSTANT_PROB,
		constant_range=CONSTANT_RANGE,
		min_depth=MIN_DEPTH,
		max_depth=MAX_DEPTH,
		y_thresh=Y_THRESH):

	funcs = []
	inp = np.zeros((ntree, nsample, nvar + 1))
	oup = np.zeros((ntree, len(operator_list)))

	for i in range(ntree):
		print('\rgenerating:', i+1, '/', ntree, end='')
		inp_i, oup_i, f_i = generate_data_instance(
			nvar=nvar, nsample=nsample,
			var_range_low=var_range_low, 
			var_range_high=var_range_high, 
			terminal_prob=terminal_prob,
			constant_prob=constant_prob,
			constant_range=constant_range,
			min_depth=min_depth, max_depth=max_depth,
			y_thresh=y_thresh,
		)
		inp[i, :, :] = inp_i
		oup[i, :] = oup_i
		funcs.append(f_i)

	print()
	dt = h5py.special_dtype(vlen=str) 
	funcs = np.array(funcs, dtype=dt)

	return inp, oup, funcs


def generate_projections(inp, ndim=NDIM):
	ntree, nsample, nvar = inp.shape
	nvar -= 1
	nproj = len(list(combinations(range(nvar), ndim)))

	inp_proj = np.zeros((ntree, nproj, nsample, ndim + 1))
	for i in range(ntree):
		for j, axes in enumerate(combinations(range(nvar), ndim)):
			inp_proj[i, j, :, :] = np.append(
				inp[i][:, axes], inp[i, :, -1][:, None], axis=1
			)
	return inp_proj


if __name__ == "__main__":
	inp, oup, funcs = generate_data()

	inp_train, oup_train, funcs_train, \
	inp_val, oup_val, funcs_val, \
	inp_test, oup_test, funcs_test = data_split(inp, oup, funcs)


	print('Saving data to', DATA_PATH)
	write_h5_file(DATA_PATH, {
		'inp_train': inp_train,
		'Y_train': oup_train,
		'funcs_train': funcs_train,
		'inp_val': inp_val,
		'Y_val': oup_val,
		'funcs_val': funcs_val,
		'inp_test': inp_test,
		'Y_test': oup_test,
		'funcs_test': funcs_test,
	})

	if NVAR > NDIM:
		inp_proj_train = generate_projections(inp_train)
		inp_proj_val = generate_projections(inp_val)
		inp_proj_test = generate_projections(inp_test)
		write_h5_file(DATA_PATH, {
			'inp_proj_train': inp_proj_train,
			'inp_proj_val': inp_proj_val,
			'inp_proj_test': inp_proj_test
		})
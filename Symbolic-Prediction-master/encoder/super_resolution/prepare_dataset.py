from itertools import combinations
import h5py
import numpy as np

from common.preset import *
from common.symbolic_tree import *
from common.write_h5_file import write_h5_file


# return a low resolution image according to the data points. Each pixel is an
# average of the z values in the bucket
def points_to_image(
	x, y, z, x_resolution=LOW_RESOLUTION[0], y_resolution=LOW_RESOLUTION[1], 
	x_low=VAR_RANGE_LOW, x_high=VAR_RANGE_HIGH, 
	y_low=VAR_RANGE_LOW, y_high=VAR_RANGE_HIGH, 
	interpolate_iter=INTERPOLATION_ITER):

	image = np.zeros((x_resolution, y_resolution))
	count = np.zeros((x_resolution, y_resolution))
	
	for i in range(len(x)):
		x_index = int((x[i] - x_low) / (x_high - x_low) * x_resolution)
		y_index = int((y[i] - y_low) / (y_high - y_low) * y_resolution)
		image[x_index, y_index] += z[i]
		count[x_index, y_index] += 1
	
	norm = np.divide(image, count, out=np.zeros_like(image), where=count!=0)
	
	# fill in pixels where there are no corresponding data points
	for k in range(interpolate_iter):
		for i in range(norm.shape[0]):
			for j in range(norm.shape[1]):
				if count[i, j] == 0:
					m_list = []
					if i > 0:
						m_list.append(norm[i-1, j])
					if j > 0:
						m_list.append(norm[i, j-1])
					if i < norm.shape[0] - 1:
						m_list.append(norm[i+1, j])
					if j < norm.shape[1] - 1:
						m_list.append(norm[i, j+1])
					norm[i, j] = np.mean(m_list)
	
	return norm


# return a high resolution image according to the symbolic tree
def tree_to_image(
	node, x_resolution=HIGH_RESOLUTION[0], y_resolution=HIGH_RESOLUTION[1], 
	x_low=VAR_RANGE_LOW, x_high=VAR_RANGE_HIGH, 
	y_low=VAR_RANGE_LOW, y_high=VAR_RANGE_HIGH):

	image = np.zeros((x_resolution, y_resolution))
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			x = (i + 0.5) / x_resolution * (x_high - x_low) + x_low
			y = (j + 0.5) / y_resolution * (y_high - y_low) + y_low
			image[i, j] = get_value(node, [x,y])
	
	return image


# remove variables that are not in the given variable set from the symbolic 
# tree. also remove all branches that do contain an extraneous variable. returns
# the trimmed symbolic tree
def retain_variables(
	node, var_set, 
	operator_list=OPERATOR_LIST, operator_args=OPERATOR_ARGS):
	
	if node.is_terminal:
		if node.variable == -1:
			return node
		elif node.variable in var_set:
			node.variable = var_set.index(node.variable)
			return node
		else:
			return None
	
	retain_results = [
		retain_variables(child, var_set) for child in node.children
	]
	node.children = list(filter(None, retain_results)) 

	if len(node.children) == 0:
		new_node = None
	elif len(node.children) < operator_args[operator_list[node.operator]]:
		new_node = node.children[0]
	else:
		new_node = node
	
	return new_node


def get_naive_encodings(
	inp, x_resolution=LOW_RESOLUTION[0], y_resolution=LOW_RESOLUTION[1], 
	x_low=VAR_RANGE_LOW, x_high=VAR_RANGE_HIGH, 
	y_low=VAR_RANGE_LOW, y_high=VAR_RANGE_HIGH, 
	interpolate_iter=INTERPOLATION_ITER):

	projected = len(inp.shape) == 4
	ntree = len(inp)

	if projected:
		nproj = inp.shape[1]
		lr = np.zeros((ntree, nproj, x_resolution, y_resolution))
		for i in range(ntree):
			for j in range(nproj):
				lr[i, j] = points_to_image(
					inp[i, j, :, 0], inp[i, j, :, 1], inp[i, j, :, 2],
					x_resolution=x_resolution, y_resolution=y_resolution, 
					x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high, 
					interpolate_iter=interpolate_iter,
				)
	else:
		lr = np.zeros((ntree, x_resolution, y_resolution))

		for i in range(ntree):
			lr[i] = points_to_image(
				inp[i, :, 0], inp[i, :, 1], inp[i, :, 2],
				x_resolution=x_resolution, y_resolution=y_resolution, 
				x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high, 
				interpolate_iter=interpolate_iter,
			)
			
	return lr

	


def get_encodings(
	inp, funcs, nvar=NVAR, ndim=NDIM,
	lr_x_resolution=LOW_RESOLUTION[0], lr_y_resolution=LOW_RESOLUTION[1], 
	hr_x_resolution=HIGH_RESOLUTION[0], hr_y_resolution=HIGH_RESOLUTION[1], 
	x_low=VAR_RANGE_LOW, x_high=VAR_RANGE_HIGH, 
	y_low=VAR_RANGE_LOW, y_high=VAR_RANGE_HIGH, 
	interpolate_iter=INTERPOLATION_ITER,
	operator_list=OPERATOR_LIST, operator_args=OPERATOR_ARGS):

	projected = len(inp.shape) == 4
	ntree = len(funcs)

	if projected:
		nproj = inp.shape[1]
		lr = np.zeros((ntree, nproj, lr_x_resolution, lr_y_resolution))
		hr = np.zeros((ntree, nproj, hr_x_resolution, hr_y_resolution))

		for i in range(ntree):
			for j, axes in enumerate(combinations(range(nvar), ndim)):

				lr[i, j] = points_to_image(
					inp[i, j, :, 0], inp[i, j, :, 1], inp[i, j, :, 2],
					x_resolution=lr_x_resolution, y_resolution=lr_y_resolution, 
					x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high, 
					interpolate_iter=interpolate_iter,
				)

				node = get_node_from_symbolic_form(funcs[i])
				node = retain_variables(
					node, axes, 
					operator_list=operator_list, operator_args=operator_args,
				)

				if node is not None:
					hr[i, j] = tree_to_image(
						node, x_resolution=hr_x_resolution, y_resolution=hr_y_resolution, 
						x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high,
					)

				hr[i, j] += np.mean(lr[i, j]) - np.mean(hr[i, j])
	else:
		lr = np.zeros((ntree, lr_x_resolution, lr_y_resolution))
		hr = np.zeros((ntree, hr_x_resolution, hr_y_resolution))

		for i in range(ntree):
			lr[i] = points_to_image(
				inp[i, :, 0], inp[i, :, 1], inp[i, :, 2],
				x_resolution=lr_x_resolution, y_resolution=lr_y_resolution, 
				x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high, 
				interpolate_iter=interpolate_iter,
			)
			node = get_node_from_symbolic_form(funcs[i])
			hr[i] = tree_to_image(
				node, x_resolution=hr_x_resolution, y_resolution=hr_y_resolution, 
				x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high,
			)

	return lr, hr


if __name__ == "__main__":
	f = h5py.File(DATA_PATH, 'r')

	Y_train = f['Y_train'][:]
	Y_val = f['Y_val'][:]
	Y_test = f['Y_test'][:]
	funcs_train = f['funcs_train']
	funcs_val = f['funcs_val']
	funcs_test = f['funcs_test']

	if NVAR == NDIM:
		lr_train, hr_train = get_encodings(f['inp_train'], funcs_train)
		lr_val, hr_val = get_encodings(f['inp_val'], funcs_val)
		lr_test, hr_test = get_encodings(f['inp_test'], funcs_test)
	else:
		lr_train, hr_train = get_encodings(f['inp_proj_train'], funcs_train)
		lr_val, hr_val = get_encodings(f['inp_proj_val'], funcs_val)
		lr_test, hr_test = get_encodings(f['inp_proj_test'], funcs_test)

	print('Saving to', SUPERR_PATH)
	write_h5_file(SUPERR_PATH, {
		'LOWR_train':lr_train,
		'HIGHR_train':hr_train,
		'Y_train':Y_train,
		'LOWR_val':lr_val,
		'HIGHR_val':hr_val,
		'Y_val':Y_val,
		'LOWR_test':lr_test,
		'HIGHR_test':hr_test,
		'Y_test':Y_test,
	})
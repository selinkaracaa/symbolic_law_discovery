import h5py
import numpy as np

from common.preset import *
from common.write_h5_file import write_h5_file
from model.MLP import MLP_Encoder

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def get_MLP_encoding(
	clf, 
	x_resolution=HIGH_RESOLUTION[0], y_resolution=HIGH_RESOLUTION[1], 
	x_low=VAR_RANGE_LOW, x_high=VAR_RANGE_HIGH, 
	y_low=VAR_RANGE_LOW, y_high=VAR_RANGE_HIGH):

	indices = np.zeros((x_resolution * y_resolution, 2))
	for i in range(x_resolution):
		for j in range(y_resolution):
			x = (i + 0.5) / x_resolution * (x_high - x_low) + x_low
			y = (j + 0.5) / y_resolution * (y_high - y_low) + y_low
			indices[i * y_resolution + j, 0] = x
			indices[i * y_resolution + j, 1] = y
	return clf.predict(indices).reshape((x_resolution, y_resolution))


def get_best_MLP_model(
	X, Y, max_iter=MLP_ENCODER_ITER, threshold=MLP_ENCODER_THRESHOLD):

	best_score = -np.inf
	best_clf = None

	i = 0

	while True:
		i += 1
		clf = MLP_Encoder()
		clf.fit(X, Y)
		s = clf.score(X, Y)

		if s > best_score:
			best_score = s
			best_clf = clf

		if i == max_iter or best_score > threshold:
			return best_clf, best_score


def encode(
	inp, 
	x_resolution=HIGH_RESOLUTION[0], y_resolution=HIGH_RESOLUTION[1], 
	x_low=VAR_RANGE_LOW, x_high=VAR_RANGE_HIGH, 
	y_low=VAR_RANGE_LOW, y_high=VAR_RANGE_HIGH, 
	max_iter=MLP_ENCODER_ITER, threshold=MLP_ENCODER_THRESHOLD):

	projected = len(inp.shape) == 4

	if projected:
		ntree, nproj = inp.shape[0], inp.shape[1]
		inp = inp.reshape((ntree * nproj, *inp.shape[2:]))

	n = len(inp)
	images = np.zeros((n, x_resolution, y_resolution))
	for i in range(n):
		clf, _ = get_best_MLP_model(
			inp[i, :, :-1], inp[i, :, -1],
			max_iter=max_iter, threshold=threshold,
		)
		images[i, :, :] = get_MLP_encoding(
			clf, 
			x_resolution=x_resolution, y_resolution=y_resolution, 
			x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high,
		)

	if projected:
		images = images.reshape((ntree, nproj, *images.shape[1:]))

	return images


if __name__ == "__main__":
	f = h5py.File(DATA_PATH, 'r')
	Y_train = f['Y_train'][:]
	Y_val = f['Y_val'][:]
	Y_test = f['Y_test'][:]

	if NVAR == NDIM:
		images_train = encode(f['inp_train'][:])
		images_val = encode(f['inp_val'][:])
		images_test = encode(f['inp_test'][:])
	else:
		images_train = encode(f['inp_proj_train'][:])
		images_val = encode(f['inp_proj_val'][:])
		images_test = encode(f['inp_proj_test'][:])

	print('Saving encoding to', MLP_ENCODER_PATH)
	write_h5_file(MLP_ENCODER_PATH, {
		'X_train':images_train,
		'Y_train':Y_train,
		'X_val':images_val,
		'Y_val':Y_val,
		'X_test':images_test,
		'Y_test':Y_test,
	})

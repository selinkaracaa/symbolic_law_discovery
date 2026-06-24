import h5py
import numpy as np
import tensorflow as tf

from common.channel import add_channel, remove_channel
from common.preset import *
from common.write_h5_file import write_h5_file
from model.super_resolution import EDSR


def encode(model, dataX, batch_size=BATCH_SIZE):
	projected = len(dataX.shape) == 4

	if projected:
		N, nproj, lr_x_resolution, lr_y_resolution = dataX.shape
		hr_x_resolution, hr_y_resolution = \
			model.compute_output_shape((1, lr_x_resolution, lr_y_resolution, 1))[1:3]
		results = np.zeros((N, nproj, hr_x_resolution, hr_y_resolution))
		for i in range(N):
			results[i] = remove_channel(model.predict(add_channel(dataX[i])))
	else:
		N, lr_x_resolution, lr_y_resolution = dataX.shape
		hr_x_resolution, hr_y_resolution = \
			model.compute_output_shape((1, lr_x_resolution, lr_y_resolution, 1))[1:3]
		results = np.zeros((N, hr_x_resolution, hr_y_resolution))
		for i in range(int(np.ceil(N / batch_size))):
			start = i * batch_size
			end = min(N, (i + 1) * batch_size)
			results[start: end] = remove_channel(model.predict(add_channel(dataX[start:end])))

	return results


if __name__ == "__main__":
	f = h5py.File(SUPERR_PATH, 'r')
	X_train = f['LOWR_train']
	X_val = f['LOWR_val']
	X_test = f['LOWR_test']

	model = EDSR()
	model.load_weights(SUPERR_ENCODER_WEIGHT_PATH)

	pred_train = encode(model, X_train)
	pred_val = encode(model, X_val)
	pred_test = encode(model, X_test)

	f.close()

	print('Saving encoding to', SUPERR_PATH)
	write_h5_file(SUPERR_PATH, {
        'X_train':pred_train,
        'X_val':pred_val,
        'X_test':pred_test,
    })

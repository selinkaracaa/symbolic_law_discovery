import numpy as np
import tensorflow as tf
import h5py

from common.preset import *
from encoder.super_resolution.run_encoder import encode
from model.super_resolution import EDSR


if __name__ == "__main__":
	f = h5py.File(SUPERR_PATH, 'r')
	X_test = f['LOWR_test']
	Y_test = f['HIGHR_test']

	model = EDSR()
	model.load_weights(SUPERR_ENCODER_WEIGHT_PATH)

	P = encode(model, X_test)
	MAE = np.abs(Y_test - P).mean()

	f.close()
	print("mean absolute error:", MAE)

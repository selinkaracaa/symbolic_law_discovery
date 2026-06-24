from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import h5py
import numpy as np
import tensorflow as tf

from common.preset import *
from encoder.super_resolution.data_loader import load_superr_data
from model.super_resolution import EDSR


def train_super_resolution(
	X_train, Y_train, X_val, Y_val, 
	batch_size=BATCH_SIZE, y_tresh=Y_THRESH, superr_lr=SUPERR_LR, 
	superr_patience=SUPERR_PATIENCE, epochs=SUPERR_EPOCHS):

	projected = len(X_train.shape) == 4

	if projected:
		N_train = X_train.shape[0] * X_train.shape[1]
		N_val = X_val.shape[0] * X_val.shape[1]
	else:
		N_train = X_train.shape[0]
		N_val = X_val.shape[0]

	steps_per_epoch = int(np.ceil(N_train / batch_size))
	validation_steps = int(np.ceil(N_val / batch_size))

	model = EDSR()
	model.compile(loss='mean_absolute_error', optimizer=Adam(lr=superr_lr))
	es = EarlyStopping(monitor='val_loss', patience=superr_patience, verbose=1)

	train_generator = load_superr_data(
		X_train, Y_train, N_train, batch_size=batch_size, y_tresh=y_tresh,
	)

	val_generator = load_superr_data(
		X_val, Y_val, N_val, batch_size=batch_size, y_tresh=y_tresh,
	)

	model.fit_generator(
		train_generator, 
		steps_per_epoch=steps_per_epoch, epochs=epochs,
		validation_data=val_generator, validation_steps=validation_steps,
		callbacks=[es], verbose=1,
	)

	return model


if __name__ == "__main__":
	f = h5py.File(SUPERR_PATH, 'r')

	X_train = f['LOWR_train']
	Y_train = f['HIGHR_train']
	X_val = f['LOWR_val']
	Y_val = f['HIGHR_val']

	model = train_super_resolution(X_train, Y_train, X_val, Y_val)
	model.save_weights(SUPERR_ENCODER_WEIGHT_PATH)
	
	f.close()
	

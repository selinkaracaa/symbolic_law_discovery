import h5py
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import numpy as np
import sys
import tensorflow as tf

from common.preset import *
from decoder.data_loader import load_decoder_data
from model.CNN import CNN
from model.MIL_Attention import MIL_Attention


def train_decoder(
	X_train, Y_train, X_val, Y_val, 
	batch_size=BATCH_SIZE, decoder_lr=DECODER_LR, 
	decoder_momentum=DECODER_MOMENTUM,
	decoder_patience=DECODER_PATIENCE, epochs=DECODER_EPOCHS):

	projected = len(X_train.shape) == 4

	if projected:
		steps_train = len(X_train)
		steps_val = len(X_val)
		model = MIL_Attention()
	else:
		steps_train = int(np.ceil(len(X_train) / batch_size))
		steps_val = int(np.ceil(len(X_val) / batch_size))
		model = CNN()

	sgd = SGD(lr=decoder_lr, momentum=decoder_momentum)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	es = EarlyStopping(monitor='val_accuracy', patience=decoder_patience, verbose=1)

	train_generator = \
		load_decoder_data(X_train, Y_train, batch_size=batch_size)
	val_generator = load_decoder_data(X_val, Y_val)

	model.fit_generator(
		train_generator, 
		steps_per_epoch=steps_train, epochs=epochs,
		validation_data=val_generator, validation_steps=steps_val,
		callbacks=[es], verbose=1,
	)

	return model


if __name__ == "__main__":
	op_index = int(sys.argv[1])
	ENCODING_PATH = SUPERR_PATH if ENCODING_SCHEME == "super resolution" else MLP_ENCODER_PATH
	f = h5py.File(ENCODING_PATH, 'r')
	X_train = f['X_train']
	Y_train = f['Y_train'][:, op_index]
	X_val = f['X_val']
	Y_val = f['Y_val'][:, op_index]

	model = train_decoder(X_train, Y_train, X_val, Y_val)

	f.close()
	model.save_weights(DECODER_WEIGHT_PATH_TEMPLATE.format(NVAR, op_index))

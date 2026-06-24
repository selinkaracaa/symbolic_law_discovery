import h5py
import keras.backend as K
import numpy as np
import tensorflow as tf

from common.channel import add_channel
from common.preset import *
from common.symbolic_tree import OPERATOR_LIST
from common.write_h5_file import write_h5_file
from model.CNN import CNN
from model.MIL_Attention import MIL_Attention


def predict_for_op(
	model, X, op_index, nvar=NVAR, 
	path_template=DECODER_WEIGHT_PATH_TEMPLATE):
	model.load_weights(path_template.format(nvar, op_index))
	predictions = np.zeros(len(X))
	for i in range(len(X)):
		batch = add_channel(X[i])
		if len(batch.shape) == 3:
			batch = np.array([batch])
		predictions[i] = model.predict_on_batch(batch)
	return predictions

def get_att_for_op(
	model, X, op_index, nvar=NVAR, 
	path_template=DECODER_WEIGHT_PATH_TEMPLATE):
	functor = K.function([model.input], [model.layers[-3].output])
	model.load_weights(path_template.format(nvar, op_index))
	att = np.zeros((X.shape[0], X.shape[1]))
	for i in range(len(X)):
		batch = add_channel(X[i])
		att[i] = functor(batch)[0][:, 0]
	return att


if __name__ == "__main__":
	ENCODING_PATH = SUPERR_PATH if ENCODING_SCHEME == "super resolution" else MLP_ENCODER_PATH
	f = h5py.File(ENCODING_PATH, 'r')

	X_train = f['X_train']
	X_val = f['X_val']
	X_test = f['X_test']


	if len(X_test.shape) == 3:
		model = CNN()
	else:
		model = MIL_Attention()

	pred_train = np.zeros((len(X_train), len(OPERATOR_LIST)))
	pred_val = np.zeros((len(X_val), len(OPERATOR_LIST)))
	pred_test = np.zeros((len(X_test), len(OPERATOR_LIST)))

	for i in range(len(OPERATOR_LIST)):
		pred_train[:, i] = predict_for_op(model, X_train, i)
		pred_val[:, i] = predict_for_op(model, X_val, i)
		pred_test[:, i] = predict_for_op(model, X_test, i)

	print('Predictions saved to', DATA_PATH)
	write_h5_file(DATA_PATH, {
		'prediction_train': pred_train,
		'prediction_val': pred_val,
		'prediction_test': pred_test,
	})
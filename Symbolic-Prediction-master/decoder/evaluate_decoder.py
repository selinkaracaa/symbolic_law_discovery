import numpy as np
import tensorflow as tf
import h5py
import pickle
from common.channel import add_channel
from common.symbolic_tree import OPERATOR_LIST
from common.preset import *
from decoder.run_decoder import predict_for_op
from model.CNN import CNN
from model.MIL_Attention import MIL_Attention
from sklearn.metrics import accuracy_score, roc_curve
from scipy.integrate import trapezoid as trapz


def evaluate_for_op(model, X, Y, op_index, nvar=NVAR):
	Y = Y[:, op_index]
	predictions = predict_for_op(model, X, op_index, nvar=nvar)
	accuracy = accuracy_score(predictions > 0.5, Y)
	fpr, tpr, thresholds = roc_curve(Y, predictions)
	return accuracy, fpr, tpr, thresholds


if __name__ == "__main__":
	ENCODING_PATH = SUPERR_PATH if ENCODING_SCHEME == "super resolution" else MLP_ENCODER_PATH
	f = h5py.File(ENCODING_PATH, 'r')
	X_test = f['X_test']
	Y_test = f['Y_test']

	if len(X_test.shape) == 3:
		model = CNN()
	else:
		model = MIL_Attention()

	acc = []
	auc = []
	roc = []

	for op_index in range(len(OPERATOR_LIST)):
		accuracy, fpr, tpr, thresholds = evaluate_for_op(model, X_test, Y_test, op_index)
		acc.append(accuracy)
		auc.append(trapz(tpr, x=fpr))
		roc.append([fpr, tpr, thresholds])

		print(OPERATOR_LIST[op_index], ':', 'acc:', accuracy, 'auc:', trapz(tpr, x=fpr))

	pickle.dump((acc, auc, roc), open(RESULT_PATH, 'wb'))
	f.close()

from sklearn.neural_network import MLPRegressor, MLPClassifier

from common.preset import *


def MLP_Encoder():
	return MLPRegressor(
		alpha=MLP_ENCODER_ALPHA,
		learning_rate="adaptive", 
		hidden_layer_sizes=MLP_ENCODER_SIZE,
		learning_rate_init=MLP_ENCODER_LR_INIT,
		early_stopping=True,
		max_iter=MLP_ENCODER_MAX_ITER
	)


def MLP_Baseline():
	return MLPClassifier(
		learning_rate="adaptive", 
		hidden_layer_sizes=MLP_BASELINE_SIZE,
		learning_rate_init=MLP_BASELINE_LR_INIT,
		verbose=True, 
		early_stopping=True
	)
import numpy as np
from common.preset import *
from common.channel import add_channel


def load_decoder_data(dataX, dataY, batch_size=BATCH_SIZE):
	if len(dataX.shape) == 4:
		return load_decoder_data_projection(dataX, dataY)
	else:
		return load_decoder_data_simple(dataX, dataY, batch_size=batch_size)
		

def load_decoder_data_simple(dataX, dataY, batch_size=BATCH_SIZE):
	N = len(dataX)
	while True:
		train_perm = np.random.permutation(N)
		for i in range(int(np.ceil(N / batch_size))):
			start = i * batch_size
			end = min(start + batch_size, N)
			indices = train_perm[start:end]
			indices.sort()
			X = dataX[indices]
			Y = dataY[indices]
			X = add_channel(X)
			yield X, Y


def load_decoder_data_projection(dataX, dataY):
	N = len(dataX)
	while True:
		train_perm = np.random.permutation(N)
		for i in train_perm:
			X = dataX[i]
			Y = dataY[i]
			X = add_channel(X)
			Y = np.ones(X.shape[0]) * Y
			yield X, Y
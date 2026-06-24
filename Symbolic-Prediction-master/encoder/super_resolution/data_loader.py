import numpy as np

from common.preset import *
from common.channel import add_channel


def load_superr_data(
	dataX, dataY, N, 
	batch_size=BATCH_SIZE, y_tresh=Y_THRESH):

	projected = len(dataX.shape) == 4

	while True:
		train_perm = np.random.permutation(N)
		for i in range(int(np.ceil(N / batch_size))):
			start = i * batch_size
			end = min(start + batch_size, N)
			indices = train_perm[start:end]
			indices.sort()

			if projected:
				X = np.zeros((len(indices), *dataX.shape[2:]))
				Y = np.zeros((len(indices), *dataY.shape[2:]))

				for i, index in enumerate(indices):
					X[i] = dataX[index // dataX.shape[1], index % dataX.shape[1]]
					Y[i] = dataY[index // dataX.shape[1], index % dataX.shape[1]]
			else:
				X = dataX[indices]
				Y = dataY[indices]

			X = add_channel(X)
			Y = add_channel(Y).clip(-y_tresh, y_tresh)
			yield X, Y
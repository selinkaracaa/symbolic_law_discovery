import numpy as np

def add_channel(X):
	return X.reshape(X.shape + (1,))

def remove_channel(X):
	return X.reshape(X.shape[:-1])
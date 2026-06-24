from common.preset import *


def data_split(*args, use_val=True):
	N = len(args[0])
	train_n = int(N * TRAIN_PROP)
	val_n = int(N * VAL_PROP)
	test = [x[train_n+val_n:] for x in args]
	if use_val:
		train = [x[:train_n] for x in args]
		val = [x[train_n:train_n+val_n] for x in args]
		return train + val + test
	else:
		train = [x[:train_n+val_n] for x in args]
		return train + test
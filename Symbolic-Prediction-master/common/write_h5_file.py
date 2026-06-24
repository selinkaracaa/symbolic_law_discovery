import h5py


def write_h5_file(fname, data):
	f = h5py.File(fname, 'a')
	for k, v in data.items():
		if k in f.keys():
			del f[k]
		f.create_dataset(k, shape=v.shape, data=v)
	f.flush()
	f.close()
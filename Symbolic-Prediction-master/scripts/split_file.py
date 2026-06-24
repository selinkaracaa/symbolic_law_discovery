import h5py
import sys

fname = sys.argv[1]
frags = int(sys.argv[2])

f = h5py.File(fname, 'r')
f_frags = [h5py.File(fname[:-3]+'-'+str(i+1)+'.h5', 'w') for i in range(frags)]

for k in f.keys():
	l = f[k].shape[0]
	for i in range(frags):
		start = round(i * l // frags)
		end = round((i+1) * l // frags)
		f_frags[i].create_dataset(k, data=f[k][start:end])

for f_frag in f_frags:
	f_frag.flush()
	f_frag.close()

f.close()
import numpy as np

SHO_A = 2.
IIG_k = 1.
IIG_V0 = 3.


def sho(k, t):
	return SHO_A * np.sin(np.sqrt(np.abs(k)) * t)

def iig(T, V):
	return IIG_k * T * np.log(np.abs(V / IIG_V0))

def fsa(r, R, h):
	return np.pi * (R + r) * np.sqrt((R - r)**2 + h**2)

def tre(vi, mi, mf, g, t):
	return vi * np.log(np.abs(mi / mf)) - g * t


def get_formula(name):
	if name == 'sho':
		return sho, 2
	elif name == 'iig':
		return iig, 2
	elif name == 'fsa':
		return fsa, 3
	elif name == 'tre':
		return tre, 5

def get_formula_variables(name):
	if name == 'sho':
		return np.array([r'$k$', r'$t$'])
	elif name == 'iig':
		return np.array([r'$T$', r'$V$'])
	elif name == 'fsa':
		return np.array([r'$r$', r'$R$', r'$h$'])
	elif name == 'tre':
		return np.array([r'$v_i$', r'$m_i$', r'$m_f$', r'$g$', r'$t$'])

def get_ground_truth(name):
	if name == 'sho':
		return 'sin(mul(sqrt(X0),X1))'.format(SHO_A)
	elif name == 'iig':
		return 'mul(X0,log(mul(X1,{:.2f})))'.format(1/IIG_V0)
	elif name == 'fsa':
		return 'mul({:.2f},mul(add(X0,X1),sqrt(add(add(add(mul(X0,X0),mul(X1,X1)),mul(X2,X2)),mul(-2,mul(X0,X1))))))'.format(np.pi)
	elif name == 'tre':
		return 'add(mul(X0,log(mul(X1,inv(X2)))),mul(-1,mul(X3,X4)))'
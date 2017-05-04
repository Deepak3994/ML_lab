import numpy as np
x1 = [0,0,1,2,2,2,1,0,0,2,0,1,1,2]
x2 = [0,0,0,1,2,2,2,1,2,1,1,1,0,1]
x3 = [0,0,0,0,1,1,1,0,1,1,1,0,1,0]
x4 = [0,1,0,0,0,1,1,0,0,0,1,1,0,1]
y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])



def partition(a):
	return {c: (a==c).nonzero()[0] for c in np.unique(a)}

def entropy(s):
	res = 0
	val, counts = np.unique(s, return_counts=True)
	freqs = counts.astype('float')/len(s)
	for p in freqs:
		if p != 0.0:
			res -= p * np.log2(p)
	return res

def mutual_information(y, x):
	res = entropy(y)
	val, counts = np.unique(x, return_counts=True)
	freqs = counts.astype('float')/len(x)
	for p, v in zip(freqs, val):
		res -= p * entropy(y[x == v])
	return res

from pprint import pprint

def is_pure(s):
	return len(set(s)) == 1

def recursive_split(x, y):
	if is_pure(y) or len(y) == 0:
		return y
	gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
	selected_attr = np.argmax(gain)
	if np.all(gain < 1e-6):
		return y
	sets = partition(x[:, selected_attr])
	res = {}
	for k, v in sets.items():
		y_subset = y.take(v, axis=0)
		x_subset = x.take(v, axis=0)
		res["x_%d = %d" % (selected_attr, k)] = recursive_split(x_subset, y_subset)

	return res

X = np.array([x1, x2]).T
pprint(recursive_split(X, y))
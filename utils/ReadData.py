import numpy as np
from random import randint

def ReadData(Filename):
	X, Y = [], []
	with open(Filename, "r") as f:
		for line in f:
			Line = line.split()
			X.append([1.0] + [float(x)for x in Line[:-1]])
			Y.append(int(Line[-1]))

	return np.array(X), np.array(Y)

def ReadData_(Filename):
	X, Y = [], []
	with open(Filename, "r") as f:
		for line in f:
			Line = line.split()
			X.append([float(x)for x in Line[:-1]])
			Y.append(int(Line[-1]))

	return np.array(X), np.array(Y)

def Bootstrap(X, Y, N):
	X_, Y_ = [], []
	for i in range(N):
		ID = randint(0, len(X)-1)
		X_.append(X[ID])
		Y_.append(Y[ID])

	return np.array(X_), np.array(Y_)
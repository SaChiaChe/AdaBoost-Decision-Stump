def CalPseudoinverse(X, Lambda):
	import numpy as np
	from numpy.linalg import pinv
	temp = X.T @ X + Lambda * np.eye(len(X[0]))
	return pinv(temp)

def CalError(W, X, Y):
	Datasize = len(X)
	ErrorCount = 0
	for i in range(Datasize):
		ErrorCount += 0 if W @ X[i] * Y[i] >= 0 else 1
	return ErrorCount / Datasize

def CalBagError(W, X, Y):
	Datasize = len(X)
	ErrorCount = 0
	for i in range(Datasize):
		temp = 0
		for w in W:
			temp += 1 if w @ X[i] * Y[i] >= 0 else -1

		ErrorCount += 0 if temp >= 0 else 1

	return ErrorCount / Datasize
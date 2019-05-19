import sys
import numpy as np
from sklearn.linear_model import Ridge
from utils.ReadData import *
from utils.Calculation import *

def RidgeRegression(X, Y, Lambda = 1):
	W = CalPseudoinverse(X, Lambda) @ X.T @ Y
	return W


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python RidgeRegression.py DataPath lambda")
		exit(0)

	Lambda = float(sys.argv[2])

	# Read Data
	X, Y = ReadData(sys.argv[1])

	# Cut train and test data
	TrainX, TestX = X[:400], X[400:]
	TrainY, TestY = Y[:400], Y[400:]

	# Run RidgeRegression
	W = RidgeRegression(TrainX, TrainY, Lambda)

	Clf = Ridge(alpha = Lambda)
	Clf.fit(TrainX, TrainY)

	# Calculate error
	E_in = CalError(W, TrainX, TrainY)
	E_out = CalError(W, TestX, TestY)
	print("My E_in:", E_in)
	print("My E_out:", E_out)

	Datasize = len(TrainX)
	EinCount = 0
	Predict = Clf.predict(TrainX)
	for i in range(len(Predict)):
		EinCount += 0 if Predict[i] * TrainY[i] >= 0 else 1
	print("Sklearn E_in:", EinCount / Datasize) 

	Datasize = len(TestX)
	EoutCount = 0
	Predict = Clf.predict(TestX)
	for i in range(len(Predict)):
		EoutCount += 0 if Predict[i] * TestY[i] >= 0 else 1
	print("Sklearn E_out:", EoutCount / Datasize)
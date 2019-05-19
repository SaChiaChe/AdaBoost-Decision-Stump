import sys
import numpy as np
from sklearn.linear_model import Ridge
from utils.ReadData import *
from utils.Calculation import *

def RidgeRegression(X, Y, Lambda = 1):
	W = CalPseudoinverse(X, Lambda) @ X.T @ Y
	return W


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Format: python RidgeRegression+Bagging.py DataPath Lambda Iteration")
		exit(0)

	Lambda = float(sys.argv[2])
	Iteration = int(sys.argv[3])

	# Read Data
	X, Y = ReadData(sys.argv[1])

	# Cut train and test data
	TrainX, TestX = X[:400], X[400:]
	TrainY, TestY = Y[:400], Y[400:]

	# Run 250 iterations
	TrackW = []
	TrackSklearnTrain = []
	TrackSklearnTest = []
	for iter in range(Iteration):
		# Bootstrap train data
		TrainX_, TrainY_ = Bootstrap(TrainX, TrainY, 400)

		# Run RidgeRegression
		W = RidgeRegression(TrainX_, TrainY_, Lambda)
		TrackW.append(W)

		Clf = Ridge(alpha = Lambda)
		Clf.fit(TrainX_, TrainY_)
		TrackSklearnTrain.append(Clf.predict(TrainX))
		TrackSklearnTest.append(Clf.predict(TestX))

	# Calculate error
	E_in = CalBagError(TrackW, TrainX, TrainY)
	E_out = CalBagError(TrackW, TestX, TestY)
	print("My E_in:", E_in)
	print("My E_out:", E_out)

	EinCount = 0
	for i in range(len(TrainX)):
		temp = 0
		for j in range(Iteration):
			temp += 1 if TrackSklearnTrain[j][i] * TrainY[i] >= 0 else -1
		EinCount += 0 if temp >= 0 else 1
	print("Sklearn E_in:", EinCount / len(TrainX)) 

	EoutCount = 0
	for i in range(len(TestX)):
		temp = 0
		for j in range(Iteration):
			temp += 1 if TrackSklearnTest[j][i] * TestY[i] >= 0 else -1
		EoutCount += 0 if temp >= 0 else 1
	print("Sklearn E_out:", EoutCount / len(TestX)) 
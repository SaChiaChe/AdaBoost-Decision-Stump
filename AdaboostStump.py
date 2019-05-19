import sys
import numpy as np
from math import log
import matplotlib.pyplot as plt

def CalUpdate(epsilon):
	return ((1 - epsilon) / epsilon)**0.5

def sign(x):
	return 1 if x >= 0 else -1

def Predict_(X, DS):
	Prediction = []
	s, i, theta = DS
	for x in X:
		temp = (x[i] - theta) * s
		Prediction.append(sign(temp))
	return Prediction

def Predict(X, DS, Alpha):
	Prediction = []
	DS = np.array(DS)
	s, i, theta = DS[:,0], DS[:,1], DS[:,2]
	for x in X:
		temp = 0.0
		for t in range(len(DS)):
			temp += sign((x[int(i[t])] - theta[t]) * s[t]) * Alpha[t]
		Prediction.append(sign(temp))
	return Prediction

def CalError(X, Y, s, i, theta, U):
	ErrorCount = 0.0
	DataSize = len(X)
	for n in range(DataSize):
		ErrorCount += 0 if (X[n][i] - theta) * s * Y[n] >= 0 else U[n]
	return ErrorCount / sum(U)

def GetAllTheta(X):
	Dim = len(X[0])
	Thetas = []
	for Feature in range(Dim):
		Thetai = []
		SortedX = sorted(X[:, Feature])
		Thetai.append(SortedX[0]-0.1)
		for i in range(1, len(SortedX)):
			Thetai.append((SortedX[i-1] + SortedX[i]) / 2)
		Thetai.append(SortedX[-1] + 0.1)
		Thetas.append(Thetai)
	return Thetas

def DecisionStump(X, Y, U):
	Thetas = GetAllTheta(X)
	Dim = len(X[0])
	TrackDS = []
	for Feature in range(Dim):
		TrackError = []
		CurThetas = Thetas[Feature]
		for theta in CurThetas:
			TrackError.append(CalError(X, Y, 1, Feature, theta, U))
			TrackError.append(CalError(X, Y, -1, Feature, theta, U))
		BestID = TrackError.index(min(TrackError))
		s = 1 if (BestID % 2) == 0 else -1
		Theta = CurThetas[int(BestID / 2)]
		TrackDS.append([s, Feature, Theta, TrackError[BestID]])
	TrackDS = np.array(TrackDS)
	BestFeature = np.argmin(TrackDS[:,3])
	return int(TrackDS[BestFeature][0]), int(TrackDS[BestFeature][1]), TrackDS[BestFeature][2], TrackDS[BestFeature][3]

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Format: python AdaboostStump.py TrainData TestData Iteration")
		exit(0)

	Iteration = int(sys.argv[3])

	# Read Data
	Train = np.genfromtxt(sys.argv[1])
	TrainX, TrainY = Train[:,:-1], [int(x) for x in Train[:,-1]]
	Test = np.genfromtxt(sys.argv[2])
	TestX, TestY = Test[:,:-1], [int(x) for x in Test[:,-1]]

	# Adaboost
	U = [1.0/len(TrainX)] * len(TrainX)
	Alpha = []
	DS = []
	TrackU = []
	for iter in range(Iteration):
		TrackU.append(list(U))
		s, i, theta, epsilon = DecisionStump(TrainX, TrainY, U)
		DS.append([s, i, theta])
		Update = CalUpdate(epsilon)
		for n in range(len(TrainX)):
			if (TrainX[n][i] - theta) * s * TrainY[n] >= 0:
				U[n] /= Update
			else:
				U[n] *= Update
		Alpha.append(log(Update))

	# Problem 13
	t = range(Iteration)
	E_in = []
	for i in t:
		PredictY = Predict_(TrainX, DS[i])
		ErrorCount = 0
		for j in range(len(PredictY)):
			ErrorCount += 0 if PredictY[j] * TrainY[j] >= 0 else 1
		E_in.append(ErrorCount / len(PredictY))
	print("E_in(g_T):", E_in[t[-1]])
	plt.figure("Problem 13")
	plt.plot(t, E_in)
	plt.title("$E_{in}(g_t)\ v.s.\ t$")
	plt.show()

	# Problem 14
	t = range(Iteration)
	E_in = []
	for i in t:
		PredictY = Predict(TrainX, DS[:i+1], Alpha[:i+1])
		ErrorCount = 0
		for j in range(len(PredictY)):
			ErrorCount += 0 if PredictY[j] * TrainY[j] >= 0 else 1
		E_in.append(ErrorCount / len(PredictY))
	print("E_in(G_T):", E_in[t[-1]])
	plt.figure("Problem 14")
	plt.plot(t, E_in)
	plt.title("$E_{in}(G_t)\ v.s.\ t$")
	plt.show()

	# Problem 15
	t = range(Iteration)
	Ut = []
	for i in t:
		Ut.append(sum(TrackU[i]))
	print("U_T:", Ut[t[-1]])
	plt.figure("Problem 15")
	plt.plot(t, Ut)
	plt.title("$U_t\ v.s.\ t$")
	plt.show()

	# Problem 16
	t = range(Iteration)
	E_out = []
	for i in t:
		PredictY = Predict(TestX, DS[:i+1], Alpha[:i+1])
		ErrorCount = 0
		for j in range(len(PredictY)):
			ErrorCount += 0 if PredictY[j] * TestY[j] >= 0 else 1
		E_out.append(ErrorCount / len(PredictY))
	print("E_out(G_T):", E_out[t[-1]])
	plt.figure("Problem 16")
	plt.plot(t, E_out)
	plt.title("$E_{out}(G_t)\ v.s.\ t$")
	plt.show()
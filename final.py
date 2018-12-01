from __future__ import print_function
import numpy as np
import math
import struct
import sys
import utilities as util
from random import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def getdata():
	trainsize = 5000
	testsize = 1000
	folder = ""

	# read training labels
	with open(folder + "train-labels.idx1-ubyte", "rb") as f:
		data = f.read(4) # magic number
		data = f.read(4) # number of samples
		training_samples = struct.unpack('>HH',data)[1]
		traininglabels = np.empty(trainsize)
		for i in range(0,trainsize):
			data = f.read(1)
			traininglabels[i] = int(struct.unpack('>B',data)[0])

	# read training images
	with open(folder + "train-images.idx3-ubyte", "rb") as f:
		data = f.read(4) # magic number
		data = f.read(4) # number of samples
		trainingsample = np.zeros(784)
		trainingsamples = np.empty((trainsize,196))
		data = f.read(4) # y dimension
		data = f.read(4) # x dimension
		for i in range(0,trainsize):
			for j in range(0,784):
				data = f.read(1)
				trainingsample[j] = util.round(struct.unpack('>B',data)[0])
			trainingsamples[i] = util.subsample(util.resize(trainingsample), 14)

	# read test labels
	with open(folder + "t10k-labels.idx1-ubyte", "rb") as f:
		data = f.read(4) # magic number
		data = f.read(4) # number of samples
		test_samples = struct.unpack('>HH',data)[1]
		testlabels = np.empty(testsize)
		for i in range(0,testsize):
			data = f.read(1)
			testlabels[i] = int(struct.unpack('>B',data)[0])

	# read test images
	with open(folder + "t10k-images.idx3-ubyte", "rb") as f:
		data = f.read(4) # magic number
		data = f.read(4) # number of samples
		testsample = np.zeros(784)
		testsamples = np.empty((testsize,196))
		data = f.read(4) # y dimension
		data = f.read(4) # x dimension
		for i in range(0,testsize):
			for j in range(0,784):
				data = f.read(1)
				testsample[j] = util.round(struct.unpack('>B',data)[0])
			testsamples[i] = util.subsample(util.resize(testsample), 14)
	
	y_train = traininglabels
	X_train = trainingsamples
	y_test = testlabels
	X_test = testsamples
	return X_train, X_test, y_train, y_test
	# return X, y

# def run(X, y):
def run(X_train, X_test, y_train, y_test, epochs, gam, C_pen):
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
	#epochs
	fin = []
	clfs_o = list()
	acc = np.zeros(10)
	#classifiers
	for k in range(10):
		clfs_o.append(SVC(gamma=gam))
	
	# fin = []
	for p in range (epochs):
		r_X_train, r_y_train = util.shuffle(X_train, y_train, 1000)
		r_X_test, r_y_test = util.shuffle(X_test, y_test, 200)

		X_train_folds = np.empty((800,196))
		y_train_folds = np.empty(800) 
		X_test_fold = np.empty((200,196))
		y_test_fold = np.empty(200) 
		k_scores = np.zeros(10)
		#folds
		for i in range(5):
			flag = False
			for k in range(5):
				if k == i:
					X_test_fold = r_X_train[k*200:k*200 + 200]
					y_test_fold = r_y_train[k*200:k*200 + 200]
				else:
					if flag == False:
						X_train_folds = r_X_train[k*200:k*200 + 200]
						y_train_folds = r_y_train[k*200:k*200 + 200]
						flag = True
					else:
						X_train_folds = np.concatenate((X_train_folds, r_X_train[k*200:k*200 + 200]), axis= 0)
						y_train_folds = np.concatenate((y_train_folds, r_y_train[k*200:k*200 + 200]), axis= 0)
			
			clfs = list()
			#train and test classifiers
			for k in range(10):
				clfs.append(SVC(gamma=gam, C=C_pen))
			for k in range(10):
				y_e_train_folds = np.empty(800)
				for j in range (len(y_train_folds)):
					if y_train_folds[j] == k:
						y_e_train_folds[j] = 1
					else:
						y_e_train_folds[j] = -1
				clfs[k].fit(X_train_folds, y_e_train_folds)
				pred = clfs[k].predict(X_test_fold)
				#print(clfs[k].get_params())
			
				y_e_test_fold = np.empty(200)
				for j in range (len(y_test_fold)):
					if y_test_fold[j] == k:
						y_e_test_fold[j] = 1
					else:
						y_e_test_fold[j] = -1
				tp = fp = tn = fn = 0.0
				for j in range (len(y_test_fold)):
					if pred[j] == y_e_test_fold[j]:
						if pred[j] == 1:
							tp += 1
						else:
							tn += 1
					else:
						if pred[j] == 1:
							fp += 1
						else:
							fn +=1
				k_scores[k] += (tn + tp) /len(y_test_fold)
				#to do precision, recall etc
				# precision = tp / (tp + fp)
				# recall = tp/(tp + fn)
		k_scores = k_scores/5.0		
		for i in range(10):
			if k_scores[i] > acc[i]:
				acc[i] = k_scores[i]
				clfs_o[i] = clfs[i]
		fin.append(k_scores)
		# print(fin)
	# acc = accuracy_score(y_test, pred, normalize=True)
	# acc = f1_score(y_test, pred, average="weighted")
	return fin

if __name__ == "__main__":
	X_train, X_test, y_train, y_test = getdata()
	print (run(X_train, X_test, y_train, y_test, 2, 0.01, 0.5))


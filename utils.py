from sklearn.metrics import classification_report
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


def get_corr_pred_index(label, dataset, model):
	'''
	Input: A given class y, 
		   dataset (X, Y), 
		   keras NN model
	output: index of the correctly predicted CSI 
	'''
	index = []
	X, Y = dataset
	for i in range(X.shape[0]):
		sample = np.expand_dims(X[i], axis=0)
		pred = model.predict_classes(sample)[0]
		if Y[i] == label:
			if  Y[i] == pred:
				index.append(i)
	return index

def get_incorr_pred_index(label, dataset, model):
	'''
	Input: A given class y, 
		   dataset (X, Y), 
		   keras NN model
	output: index of the incorrectly predicted CSI
	'''
	index = []
	X, Y = dataset
	for i in range(X.shape[0]):
		sample = np.expand_dims(X[i], axis=0)
		pred = model.predict_classes(sample)[0]
		if Y[i] == label:
			if  Y[i] != pred:
				index.append(i)
	return index

def get_same_pred_index(label, dataset, model):
	'''
	Input: A given class y, 
		   dataset (X, Y), 
		   keras NN model
	output: index of the CSI predicted as y
	'''	
	index = []
	X, Y = dataset
	for i in range(X.shape[0]):
		sample = np.expand_dims(X[i], axis=0)
		pred = model.predict_classes(sample)[0]
		if pred == label:
			index.append(i)
	return index

def get_same_class_index(y, dataset):
	'''
	Input: A given class y, 
		   dataset (X, Y), 
		   keras NN model
	output: index of the correctly predicted 
	'''
	index = []
	X, Y = dataset
	for i in range(X.shape[0]):
		if Y[i] == y:
			index.append(i)

	return index


def compute_variance(index, relevances, axis):
	'''
	label: assign a class to compute
	dataset: (X, Y) NumPy Array
	model: trained keras model
	relevances: relevance score (whole) according to dataset and model 
	axis: 0 = cross samples, 1 = cross channels
	'''
	rs=[]
	
	for i in index:
		rs.append(relevances[i])

	rs=np.array(rs)
	std=np.std(rs, axis=axis)
	avg=np.mean(rs, axis=axis)
	std=np.around(std, decimals=5)
	avg=np.around(avg, decimals=5)
	std=std.tolist()
	avg=avg.tolist()
	val=[]
	for i in range(len(avg)):
		if avg[i]!=0:
			val.append(std[i]/avg[i])
		else:
			val.append(0)	
	return average(val)



def average(lst): 
    return round(sum(lst)/len(lst), 2) 

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def minmaxscale(data, feature_range):
    scaler = MinMaxScaler(feature_range)
    scaler.fit(data) 
    normalized = scaler.transform(data)

    return normalized  

def scale_samples(x, out_range):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])

    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2      


#if __name__ == '__main__':


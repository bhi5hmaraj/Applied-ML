 
import numpy as np
import load_mnist
import warnings

'''
Weights are assumed to be modified to incorporate bias
ie. X = 1 + x0 + x1 + ...
'''
# weights = 1 x (n + 1) , X = m x (n + 1)

def sigmoid(x):
	return 1 / (1 + (1 / np.exp(x)))

def predict(weights, X):
	# print("Product ================")
	
	prediction = sigmoid(X @ weights.T)
	assert all(prediction <= 1)
	return prediction


def cost(weights, X, y):
	pred = predict(weights, X)
	s1 = y * np.log(pred)
	s2 = (1 - y) * np.log(1 - pred)


	return -np.sum(s1 + s2)


def compute_gradients(weights, X, y):
	weights_n = (predict(weights, X) - y).T @  X
	return weights_n

def learn_weights(X, y, rate= 4e-5, iters=1000, thresh=0.1):
	weights = np.random.uniform(low=1e-4, high=1e-6, size=(1,X.shape[1]))
	
	for _ in range(iters):
		grad = compute_gradients(weights, X, y)
		weights_n = weights - rate * grad
		c_old = cost(weights, X, y)
		c_new = cost(weights_n, X, y)
		
		print("Iteration : %d , Old cost = %f , New cost = %f" % (_, c_old, c_new))
		
		if abs(c_old - c_new) < thresh:
			break

		weights = weights_n

	return weights

def train_test_split(num=5):

	train_img    = load_mnist.read_idx('train-img.data')
	train_labels = load_mnist.read_idx('train-labels.data')
	test_img 	 = load_mnist.read_idx('test-img.data')
	test_labels  = load_mnist.read_idx('test-labels.data')

	train_img = np.reshape(train_img, (train_img.shape[0], -1))
	test_img  = np.reshape(test_img, (test_img.shape[0], -1))

	# normalize the data https://stackoverflow.com/questions/35419882/cost-function-in-logistic-regression-gives-nan-as-a-result

	mu = np.mean(train_img, 0)
	sig = np.std(train_img, 0)
	train_img = np.nan_to_num((train_img - mu) / sig)
	test_img  = np.nan_to_num((test_img - mu) / sig)

	

	X_train = train_img[train_labels == num]
	X_test  = test_img[test_labels == num]

	

	l_train = X_train.shape[0]
	l_test  = X_test.shape[0]

	X_train_extra = train_img[train_labels != num][:l_train, :]
	X_test_extra  = test_img[test_labels != num][:l_test, :]

	X_train =  np.append(X_train, X_train_extra, axis=0)
	X_test  =  np.append(X_test, X_test_extra, axis=0)

	print("X_train : %s , X_test : %s " % (X_train.shape, X_test.shape))

	y_train = np.array([1] * l_train + [0] * l_train).reshape(2 * l_train, 1)
	y_test  = np.array([1] * l_test  + [0] * l_test).reshape(2 * l_test, 1)

	print("y_train : %s , y_test : %s " % (y_train.shape, y_test.shape))

	return (X_train, y_train, X_test, y_test)


def check_accuracy(X, y, weights, thresh=0.5):
	pred = predict(weights, X)
	pred[pred >= thresh] = 1
	pred[pred <  thresh] = 0
	return sum(np.logical_not(np.logical_xor(pred, y))) / len(y)

def main():

	X_train, y_train, X_test, y_test = train_test_split()
	X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis=1)
	X_test  = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

	# print("X_train : %s , X_test : %s " % (X_train[0, X_test.shape))

	print("Learning weights ...")
	w = learn_weights(X_train, y_train)
	# print("Weights learned : %s!" % w)
	# print(X_test[0])
	# print(X_test @ w.T)
	print("Accuracy in test set = %f" % check_accuracy(X_test, y_test, w))



main()
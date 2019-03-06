import numpy as np
import load_mnist


def train_test_split(num=5):

	train_img    = load_mnist.read_idx('train-img.data')
	train_labels = load_mnist.read_idx('train-labels.data')
	test_img 	 = load_mnist.read_idx('test-img.data')
	test_labels  = load_mnist.read_idx('test-labels.data')

	train_img = np.reshape(train_img, (train_img.shape[0], -1))
	test_img  = np.reshape(test_img, (test_img.shape[0], -1))
	
	return (train_img, train_labels, test_img, test_labels)



def sigmoid(x):
	return 1 / (1 + (1 / np.exp(x)))


# weights : L x (inp + 1) x (out)

def get_activations(weights, X):
	activation = sigmoid(np.append(X, np.ones((X.shape[0], 1))) @ weights[0])
	ret = np.array(X)

	for i in range(1, len(weights)):
		ret = np.append(ret, activation, axis=0)
		activation = np.append(activation, np.ones((activation.shape[0], 1)), axis=1)
		activation = sigmoid(activation @ weights[i])

	return ret

def predict(weights, X):
	activation = sigmoid(np.append(X, np.ones((X.shape[0], 1))) @ weights[0])

	for i in range(1, len(weights)):
		activation = np.append(activation, np.ones((activation.shape[0], 1)), axis=1)
		activation = sigmoid(activation @ weights[i])

	return activation

def cost(weights, X, y):
	pred = predict(weights, X)
	return (-1/len(X)) * sum(y * np.log(pred) + (1 - y) * np.log(pred))



# ref : https://en.wikipedia.org/wiki/Backpropagation

def compute_gradients(weights, X, y):
	gradients = np.zeros_like(weights)
	
	for i in range(len(X)):
		activations = get_activations(weights, X[i])

		delta = (activations[-1] - y[i]).T
		gradients[-1] += (delta @ np.append(activations[-2], [1])).T

		for j in range(len(weights) - 2, 0, -1):
			delta = np.delete(weights[j + 1] @ delta, -1, axis=0) * (activations[j + 1] * (1 - activations[j + 1])).T
			gradients[j] += (delta @ np.append(activations[j], [1])).T

	return gradients	


def learn_weights(X, y, rate= 4e-5, iters=300, thresh=0.1):
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

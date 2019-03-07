import numpy as np
import load_mnist

TAKE = 30000

def train_test_split():

	train_img    = load_mnist.read_idx('train-img.data')[:TAKE]
	train_labels = load_mnist.read_idx('train-labels.data')[:TAKE]
	test_img 	 = load_mnist.read_idx('test-img.data')
	test_labels  = load_mnist.read_idx('test-labels.data')

	train_img = np.reshape(train_img, (train_img.shape[0], -1))	# flatten
	test_img  = np.reshape(test_img, (test_img.shape[0], -1))
	

	mu  = np.mean(train_img, 0)
	sig = np.std(train_img, 0)
	train_img = np.nan_to_num((train_img - mu) / sig)
	test_img  = np.nan_to_num((test_img - mu) / sig)

	oh_test = np.zeros((len(test_labels), 10))
	oh_train = np.zeros((len(train_labels), 10))

	oh_test[np.arange(len(oh_test)), test_labels] = 1
	oh_train[np.arange(len(oh_train)), train_labels] = 1
	

	return (train_img, oh_train, test_img, oh_test)





def sigmoid(x):
	# print(np.max(x))
	return 1 / (1 + np.exp(-x))


# weights : L x (inp + 1) x (out)

def get_activations(weights, X):
	X = np.append(X, [1]).reshape(1, len(X) + 1)
	# print(weights[0].shape)
	# print(X.shape)

	activation = sigmoid(X @ weights[0])
	ret = [X]

	for i in range(1, len(weights)):
		ret.append(activation)
		activation = np.append(activation, np.ones((activation.shape[0], 1)), axis=1)
		activation = sigmoid(activation @ weights[i])

	ret.append(activation)

	return ret

def predict(weights, X):
	X = np.append(X, np.ones((len(X), 1)) , axis=1)
	activation = sigmoid(X @ weights[0])

	for i in range(1, len(weights)):
		activation = np.append(activation, np.ones((activation.shape[0], 1)), axis=1)
		activation = sigmoid(activation @ weights[i])

	return activation

def cost(weights, X, y):
	pred = predict(weights, X)
	return (-1/len(X)) * np.sum(y * np.log(pred) + (1 - y) * np.log(pred))



# ref : https://en.wikipedia.org/wiki/Backpropagation

def compute_gradients(weights, X, y):
	gradients = [np.zeros_like(w) for w in weights]
	# print(gradients)
	# print("gradients shape : {}".format([x.shape for x in gradients]))
		
	for i in range(len(X)):
		activations = get_activations(weights, X[i])
		
		# print("activations shape : {}".format([x.shape for x in activations]))
		
		delta = (activations[-1] - y[i]).T
		# print(fuck.shape)
		gradients[-1] += (delta @ np.append(activations[-2], [1]).reshape(1, activations[-2].shape[1] + 1)).T

		for j in range(len(weights) - 2, 0, -1):
			delta = np.delete(weights[j + 1] @ delta, -1, axis=0) * (activations[j + 1] * (1 - activations[j + 1])).T
			gradients[j] += (delta @ np.append(activations[j], [1]).reshape(1, activations[j].shape[1] + 1)).T

	gradients = [g / len(X) for g in gradients]

	# check_gradients(gradients, weights, X, y)

	return gradients	

def check_gradients(gradients, weights, X, y,eps=1e-4):

	error = 1e-4
	print("Checking gradients ...")

	for k in range(len(weights)):		
		for i in range(gradients[k].shape[0]):
			for j in range(gradients[k].shape[1]):

				weights[k][i][j] += eps
				c1 = cost(weights, X, y)
				
				weights[k][i][j] -= 2 * eps
				c2 = cost(weights, X, y)
				
				grad_check = (c1 - c2) / (2 * eps)
				weights[k][i][j] += eps
				e = abs(grad_check - gradients[k][i][j])
				
				if e > 0:
					print("Error in gradients = %f" % (e))
				
				assert e <= error


def learn_weights(X, y, rate= 1e-2, iters=300, thresh=0.01, layers=[128, 64, 32, 10]):
	weights = [np.random.uniform(low=1e-4, high=1e-3, size=(X.shape[1] + 1, layers[0]))]


	for i in range(1, len(layers)):
		weights.append(np.random.uniform(low=1e-4, high=1e-3, size=(layers[i - 1] + 1, layers[i])))

	print("weights shape : {}".format([x.shape for x in weights]))

	# WTF !! why should I add the grad ??

	for _ in range(iters):
		print("Computing gradients ...")
		grad = compute_gradients(weights, X, y)
		weights_n = [weights[i] + rate * grad[i] for i in range(len(weights))]

		c_old = cost(weights, X, y)
		c_new = cost(weights_n, X, y)
		
		print(c_old.shape)

		print("Iteration : %d , Old cost = %f , New cost = %f" % (_, c_old, c_new))
		
		if abs(c_old - c_new) < thresh:
			break

		weights = weights_n

	return weights




def check_accuracy(X, y, weights):
	pred = predict(weights, X)

	# https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0

	pred = (pred == pred.max(axis=1)[:,None]).astype(int)
	return sum(np.all(np.logical_not(np.logical_xor(pred,y)), axis=1)) / len(y)

def main():

	X_train, y_train, X_test, y_test = train_test_split()

	print("X_train : %s , X_test : %s , y_train : %s" % (X_train.shape, X_test.shape, y_train.shape))

	print("Learning weights ...")
	w = learn_weights(X_train, y_train)
	# print("Weights learned : %s!" % w)
	# print(X_test[0])
	# print(X_test @ w.T)
	print("Accuracy in test set = %f" % check_accuracy(X_test, y_test, w))



main()
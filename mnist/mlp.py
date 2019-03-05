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



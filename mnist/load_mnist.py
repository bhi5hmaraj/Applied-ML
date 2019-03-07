# https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40

""" A function that can read MNIST's idx file format into numpy arrays.

    The MNIST data files can be downloaded from here:
    
    http://yann.lecun.com/exdb/mnist/

    This relies on the fact that the MNIST dataset consistently uses
    unsigned char types with their data segments.
"""

import struct

import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape) 


def train_test_split(num=5):

	train_img    = load_mnist.read_idx('train-img.data')
	train_labels = load_mnist.read_idx('train-labels.data')
	test_img 	 = load_mnist.read_idx('test-img.data')
	test_labels  = load_mnist.read_idx('test-labels.data')

	train_img = np.reshape(train_img, (train_img.shape[0], -1))	# flatten
	test_img  = np.reshape(test_img, (test_img.shape[0], -1))
	

	mu  = np.mean(train_img, 0)
	sig = np.std(train_img, 0)
	train_img = np.nan_to_num((train_img - mu) / sig)
	test_img  = np.nan_to_num((test_img - mu) / sig)

	test_labels  = np.zeros((len(test_labels), 10))[np.arange(len(test_labels)), test_labels]
	train_labels = np.zeros((len(train_labels), 10))[np.arange(len(train_labels)), train_labels]

	return (train_img, train_labels, test_img, test_labels)


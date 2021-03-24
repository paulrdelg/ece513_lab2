# Standard Python Packages
import os
import platform
import sys

# Common Third-Party Packages
import matplotlib.pyplot as plt
import numpy as np
import PIL
from sklearn.decomposition import PCA

# Custom
import lab1

def main():
	# Load matrix of 1D column vectors (X)
	X = np.load('mydata/X.npy')
	print('X:   ', X.shape)
	
	# Retrieve Dimensions
	N = X.shape[0]
	M = X.shape[1]
	print('N:   ', N)
	print('M:   ', M)
	
	# Load SVD
	U = np.load('mydata/U.npy')
	Sigma = np.load('mydata/Sigma.npy')
	VT = np.load('mydata/VT.npy')
	print('U:   ', U.shape)
	print('S:   ', Sigma.shape)
	print('VT:  ', VT.shape)
	
	# Determine mean image (m)
	m = np.load('mydata/m.npy')
	print('m:   ', m.shape)
	
	# compute mean centered matrix (W)
	W = np.load('mydata/W.npy')
	WT = W.transpose()
	print('W:   ', W.shape)
	print('WT:  ', WT.shape)
	
	# Compute Covariance (C)
	C = np.load('mydata/C.npy')
	
	# Eigenvalues & Eigenvectors
	fp = 'mydata/eigenvalues.npy'
	eigenvalues = np.load(fp)
	pca = np.load('mydata/pca.npy')
	print(pca)
	print(pca.shape)
	
	# Reconstruct with 4th pca
	a = np.zeros(X.shape[0])
	print(a.shape)
	count = 0
	for i in (0, 3):
		w = W[i]
		a = a + np.matmul(W[i], VT[i])
		count = count + 1
	a = a.reshape((304, 200))
	lab1.showImage(a)
	
	return 0

if __name__ == "__main__":
	if platform.python_version_tuple()[0] == 3 and platform.python_version_tuple()[1] < 9:
		print('ERROR: Need Python 3.9.X to run')
	else:
		main()

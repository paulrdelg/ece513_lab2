# Standard Python Packages
import os
import platform
import sys

# Common Third-Party Packages
import matplotlib.pyplot as plt
import numpy as np
import PIL
from sklearn.decomposition import PCA

def extractFileNames(dirName):
	onlyfiles = [f for f in os.listdir(dirName) if os.path.isfile(os.path.join(dirName, f))]
	return onlyfiles

def definePrefix(subjectNum):
	prefix = 'subject'
	if subjectNum < 10:
		prefix = prefix + '0' + str(subjectNum)
	else:
		prefix = prefix + str(subjectNum)
	return prefix

def splitSubjects(flist):
	subjects = []
	
	subjectNum = 1
	subjectFiles = []
	for f in flist:
		
		currentPrefix = definePrefix(subjectNum)
		currentSubject = f.startswith(currentPrefix)
		
		prefixSubject = 'subject'
		suffix = '.gif'
		
		if currentSubject:
			# strip file gunk
			feature = f.removeprefix(prefixSubject).removesuffix(suffix)
			#print(feature + ' is current for ' + str(subjectNum))
			
			# add feature to current subject
			subjectFiles.append(feature)
		else:
			# save previous subject
			subjects.append(subjectFiles)
			
			# define next subject
			subjectNum = subjectNum + 1
			newPrefix = definePrefix(subjectNum)
			currentSubject = f.startswith(newPrefix)
			
			# strip file gunk
			feature = f.removeprefix(prefixSubject).removesuffix(suffix)
			#print(feature + ' is new for ' + str(subjectNum))
			
			# create new subject
			subjectFiles = [feature]
	
	subjects.append(subjectFiles)
	
	return subjects

def getSubjectFeatures(subjects, subjectNumber):
	subject = subjects[subjectNumber - 1]
	return subject

def readImage(filePath):
	# read img
	img = plt.imread(filePath)
	return img

def loadData(dirPath, subjects):
	# Initialize array to be returned
	pltImagesArray = []
	
	for subject in subjects:
		for feature in subject:
			filepath = dirPath + '/subject' + feature + '.gif'
			im = readImage(filepath)
			pltImagesArray.append(im)
	
	return pltImagesArray

def trim(img, side, bot):
	h = img.shape[0]
	w = img.shape[1]
	img = img[side:h-side][bot:]
	#yImg = PIL.Image.fromarray(img)
	#yImg.show()
	#exit()
	return img

def showImage(npv):
	img = PIL.Image.fromarray(npv)
	img.show()

def processed():
	# Load Data
	yaleFacesPath = './yalefaces'
	features = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
	flist = extractFileNames(yaleFacesPath)
	subjects = splitSubjects(flist)
	pltImagesArray = loadData(yaleFacesPath, subjects)
	
	# Trim & Convert to numpy type
	list_of_img_nparray = []
	for pltImage in pltImagesArray:
		img = np.asarray(pltImage)
		img = trim(img, 20, 13)
		list_of_img_nparray.append(img)
	
	# Get dimensions
	N = list_of_img_nparray[0].shape[0] * list_of_img_nparray[0].shape[1]
	M = 15;#len(list_of_img_nparray)
	print('N:   ', N)
	print('M:   ', M)
	
	# Construct matrix of 1D column vectors (X)
	X = np.zeros((N, M))
	count = 0
	for img in list_of_img_nparray:
		if count == 10 or count == 21 or count == 32 or count == 43 or count == 54 or count == 65 or count == 76 or count == 87 or count == 98 or  count == 109 or count == 120 or count == 131 or count == 142 or count == 153 or count == 164:
			x = np.asarray(img).flatten()
			x = x.reshape((x.shape[0], 1))
			X[:, count] = x.flatten()
			count = count + 1
	np.save('mydata/X', X)
	print('X:   ', X.shape)
	
	# Use Numpy to compute SVD
	U, Sigma, VT = np.linalg.svd(X)
	np.save('mydata/U', X)
	np.save('mydata/Sigma', Sigma)
	np.save('mydata/VT', VT)
	print('U:   ', U.shape)
	print('S:   ', Sigma.shape)
	print('VT:  ', VT.shape)
	
	# Determine mean image (m)
	x_sum = np.zeros((1,N))
	for x in X.transpose():
		x_sum = x_sum + x
	m = (x_sum/M).transpose()
	np.save('mydata/m', m)
	print('m:   ', m.shape)
	
	# compute mean centered matrix (W)
	W = np.zeros((N, M))
	print('W:   ', W.shape)
	count = 0
	for x in X.transpose():
		w = x.flatten() - m.flatten()
		W[:, count] = w
		count = count + 1
	WT = W.transpose()
	np.save('mydata/W', W)
	print('W:   ', W.shape)
	print('WT:  ', WT.shape)
	
	# Compute Covariance (C)
	C = np.matmul(W, WT, dtype=np.float64)
	np.save('mydata/C', C)
	print('C:   ', C.shape)
	
	# Eigenvalues & Eigenvectors
	w, v = np.linalg.eig(C)
	print(w)
	print(w.shape)
	w2 = -np.sort_complex(-w)
	print(w2)
	print(w2.shape)
	w3 = w2[np.logical_not(np.isnan(w2))]
	np.save('mydata/eigenvalues', w3)
	
	fp = 'mydata/eigenvalues.npy'
	eigenvalues = np.load(fp)
	eigensum = 0
	for c in eigenvalues:
		eigensum = eigensum + c
	
	abssum = abs(eigensum)
	
	csum = 0
	count = 0
	kth = 0
	for c in eigenvalues:
		csum = csum + abs(c)
		tv = csum / abssum
		if tv > 0.95:
			kth = count
			break
		count = count + 1
	
	# Create kth eigenvalues
	kpc = np.zeros(kth)
	count = 0
	for i in eigenvalues:
		if count < kth:
			kpc[count] = eigenvalues[count]
		count = count + 1
	
	np.save('mydata/pca', kpc)
	
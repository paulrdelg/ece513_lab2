# Standard Python Packages
import os
import platform
import sys

# Common Third-Party Packages
import cv2
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import PIL
from sklearn.decomposition import PCA
import scipy
#from scipy import fftpack # Legacy

def showPepperOCV(title, img):
	# Output img with window name as 'image'
	cv2.imshow(title, img)
	# Maintain output window utill
	# user presses a key
	cv2.waitKey(0)
	# Destroying present windows on screen
	cv2.destroyAllWindows()

def showPepperMPL(img):
	# Output Images
	plt.imshow(img)

def showPepperPIL(img):
	# Output Images
	img.show()
	
	# prints format of image
	print(img.format)
	
	# prints mode of image
	print(img.mode)

# implement 2D DCT
def dct2(a, n):
	return scipy.fft.dct( scipy.fft.dct(a.T, n=n, norm='ortho').T, n=n, norm='ortho')

# implement 2D IDCT
def idct2(a, n):
	return scipy.fft.idct(scipy.fft.idct(a.T, n=n, norm='ortho').T, n=n, norm='ortho')

# Standard Python Packages
import math
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
from scipy import fftpack

# Custom
import lab2

def main():
	# Read image file
	filepath = 'mydata/01_copy.bmp'
	title = 'Copy of Pepper Image (Uncompressed)'
	
	# Read image
	#img = cv2.imread(filepath)
	img = mpl.image.imread(filepath)
	#img = PIL.Image.open(filepath)
	
	# Convert PIL image to NumPy array
	img = np.asarray(img)
	
	# Convert to grayscale by getting one of the channel values (all seem the same)
	img = img[:,:,0]
	
	# Save 2-D version
	imgPIL = PIL.Image.fromarray(img)
	imgPIL.save('mydata/02_grayscale.bmp')
	
	
	# Calculate mean
	total = 0
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			total = total + img[i][j]
	u = total/(img.shape[0]*img.shape[1])
	total = 0
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			total = total + pow((img[i][j] - u), 2)
	var_o = total/(img.shape[0]*img.shape[1])
	
	# Define block sizes
	block_sizes = [1088, 1024, 960, 896, 832, 768, 704, 640, 576, 512, 384, 256, 128, 96, 64, 48, 32, 28, 24, 16, 8]
	
	
	list_var_e = []
	for k in block_sizes:
		print(k)
		imgCompressed = lab2.dct2(img, k)
		imgFixedpoint = imgCompressed.astype(np.uint8)
		imgPIL = PIL.Image.fromarray(imgFixedpoint)
		imgPIL.convert("L")
		filepath = 'mydata/04_compressed_' + str(k) + '.bmp'
		imgPIL.save(filepath)
		imgReconstructed = lab2.idct2(imgCompressed, k)
		total = 0
		for i in range(0, imgReconstructed.shape[0]):
			for j in range(0, imgReconstructed.shape[1]):
				total = total + imgReconstructed[i][j]
		u = total/(imgReconstructed.shape[0] * imgReconstructed.shape[1])
		total = 0
		for i in range(0, imgReconstructed.shape[0]):
			for j in range(0, imgReconstructed.shape[1]):
				total = total + pow((imgReconstructed[i][j] - u), 2)
		var_e = total/(imgReconstructed.shape[0]*imgReconstructed.shape[1])
		list_var_e.append(var_e)
		imgFixedpoint = imgReconstructed.astype(np.uint8)
		imgPIL = PIL.Image.fromarray(imgFixedpoint)
		imgPIL.convert("L")
		filepath = 'mydata/05_reconstructed_' + str(k) + '.bmp'
		imgPIL.save(filepath)
	
	list_snr = []
	for v in list_var_e:
		x = var_o / v
		snr = 10 * math.log(x, 10)
		list_snr.append(snr)
	
	# plotting the points
	block_sizes.reverse()
	list_snr.reverse()
	plt.plot(block_sizes, list_snr)
	# naming the x axis
	plt.xlabel('k (block size)')
	# naming the y axis
	plt.ylabel('SNR')
	# giving a title to my graph
	plt.title('SNR for various block sizes')
	# function to show the plot
	plt.show()
	
	plt.savefig('mydata/06_snr_vs_k')
	
	
	# DWT Haar level 2 and 3
	coeffs2h = pywt.wavedec2(img, 'haar', level=2)
	coeffs3h = pywt.wavedec2(img, 'haar', level=3)
	
	# Save Haar Level2 Coefficients
	cA, cB, cC = coeffs2h
	
	filepath = 'mydata/07_haar_level2_coeff1.bmp'
	cA = np.array(cA)
	cA = cA.astype(np.uint8)
	imgPIL = PIL.Image.fromarray(cA)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_haar_level2_coeff2.bmp'
	cB = np.array(cB)
	cB = cB.astype(np.uint8)
	print(cB.shape)
	cB = cB.reshape((280,280,3))
	imgPIL = PIL.Image.fromarray(cB)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_haar_level2_coeff3.bmp'
	cC = np.array(cC)
	cC = cC.astype(np.uint8)
	print(cC.shape)
	cC = cC.reshape((560,560,3))
	imgPIL = PIL.Image.fromarray(cC)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	# Save Haar Level3 Coefficients
	coeffA, coeffB, coeffC, coeffD = coeffs3h
	
	filepath = 'mydata/07_haar_level3_coeff1.bmp'
	coeffA = np.array(coeffA)
	coeffA = coeffA.astype(np.uint8)
	imgPIL = PIL.Image.fromarray(coeffA)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_haar_level3_coeff2.bmp'
	coeffB = np.array(coeffB)
	coeffB = coeffB.astype(np.uint8)
	coeffB = coeffB.reshape((140,140,3))
	imgPIL = PIL.Image.fromarray(coeffB)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_haar_level3_coeff3.bmp'
	coeffC = np.array(coeffC)
	coeffC = coeffC.astype(np.uint8)
	coeffC = coeffC.reshape((280,280,3))
	imgPIL = PIL.Image.fromarray(coeffC)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_haar_level3_coeff4.bmp'
	coeffD = np.array(coeffD)
	coeffD = coeffD.astype(np.uint8)
	coeffD = coeffD.reshape((560,560,3))
	imgPIL = PIL.Image.fromarray(coeffD)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	# Reconstruct Haar
	re2h = pywt.waverec2(coeffs2h, 'haar')
	re3h = pywt.waverec2(coeffs3h, 'haar')
	
	# Convert Haar to numpy
	re2h = np.array(re2h)
	re3h = np.array(re3h)
	
	# Convert to unsigned 8-bit
	re2h = re2h.astype(np.uint8)
	re3h = re3h.astype(np.uint8)
	
	# Save reconstructed files
	filepath = 'mydata/08_haar_level2_reconstructed.bmp'
	imgPIL = PIL.Image.fromarray(re2h)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/08_haar_level3_reconstructed.bmp'
	imgPIL = PIL.Image.fromarray(re3h)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	# DWT for DB wavelet
	coeffs2d = pywt.wavedec2(img, 'db2', level=2)
	coeffs3d = pywt.wavedec2(img, 'db2', level=3)
	
	# Save DB2 Level2 Coefficients
	coeff1, coeff2, coeff3 = coeffs2d
	
	filepath = 'mydata/07_db2_level2_coeff1.bmp'
	coeff1 = np.array(coeff1)
	coeff1 = coeff1.astype(np.uint8)
	imgPIL = PIL.Image.fromarray(coeff1)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_db2_level2_coeff2.bmp'
	coeff2 = np.array(coeff2)
	coeff2 = coeff2.astype(np.uint8)
	print(coeff2.shape)
	coeff2 = coeff2.reshape((282,282,3))
	imgPIL = PIL.Image.fromarray(coeff2)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_db2_level2_coeff3.bmp'
	coeff3 = np.array(coeff3)
	coeff3 = coeff3.astype(np.uint8)
	print(coeff3.shape)
	coeff3 = coeff3.reshape((561,561,3))
	imgPIL = PIL.Image.fromarray(coeff3)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	# Save Haar Level3 Coefficients
	coeffA, coeffB, coeffC, coeffD = coeffs3d
	
	filepath = 'mydata/07_db2_level3_coeff1.bmp'
	coeffA = np.array(coeffA)
	coeffA = coeffA.astype(np.uint8)
	print(coeffA.shape)
	imgPIL = PIL.Image.fromarray(coeffA)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_db2_level3_coeff2.bmp'
	coeffB = np.array(coeffB)
	coeffB = coeffB.astype(np.uint8)
	print(coeffB.shape)
	coeffB = coeffB.reshape((142,142,3))
	imgPIL = PIL.Image.fromarray(coeffB)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_db2_level3_coeff3.bmp'
	coeffC = np.array(coeffC)
	coeffC = coeffC.astype(np.uint8)
	print(coeffC.shape)
	coeffC = coeffC.reshape((282,282,3))
	imgPIL = PIL.Image.fromarray(coeffC)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/07_db2_level3_coeff4.bmp'
	coeffD = np.array(coeffD)
	coeffD = coeffD.astype(np.uint8)
	print(coeffD.shape)
	coeffD = coeffD.reshape((561,561,3))
	imgPIL = PIL.Image.fromarray(coeffD)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	# Reconstruct
	re2d = pywt.waverec2(coeffs2d, 'db2')
	re3d = pywt.waverec2(coeffs3d, 'db2')
	
	re2d = np.array(re2d)
	re3d = np.array(re3d)
	
	# Convert to unsigned 8-bit
	re2d = re2d.astype(np.uint8)
	re3d = re3d.astype(np.uint8)
	
	# Save reconstructed files
	filepath = 'mydata/08_db2_level2_reconstructed.bmp'
	imgPIL = PIL.Image.fromarray(re2d)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	filepath = 'mydata/08_db2_level3_reconstructed.bmp'
	imgPIL = PIL.Image.fromarray(re3d)
	imgPIL.convert("L")
	imgPIL.save(filepath)
	
	return 0

if __name__ == "__main__":
	if platform.python_version_tuple()[0] == 3 and platform.python_version_tuple()[1] < 9:
		print('ERROR: Need Python 3.9.X to run')
	else:
		main()

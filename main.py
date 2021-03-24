# Standard Python Packages
import os
import platform
import sys

# Common Third-Party Packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from sklearn.decomposition import PCA

# Custom
import lab2

def main():
	# Read image file
	filepath = 'pepper_copy.PNG'
	title = 'Copy of Pepper Image (Uncompressed)'
	
	# Read image
	#img = cv2.imread(filepath)
	#img = mpl.image.imread(filepath)
	img = PIL.Image.open(filepath)
	
	# Define block sizes
	block_sizes = [8, 16]
	
	
	
	return 0

if __name__ == "__main__":
	if platform.python_version_tuple()[0] == 3 and platform.python_version_tuple()[1] < 9:
		print('ERROR: Need Python 3.9.X to run')
	else:
		main()

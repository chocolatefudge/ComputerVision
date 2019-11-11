import numpy as np
import scipy.signal as signal
import math
from scipy.integrate import quad

def reshape(data):
	return data.reshape((data.shape[0], 1))

def gaussian(sigma):
	"""Computes (3, 1) array corresponding to a Gaussian filter.
	Normalisation is not required.

	Args:
		sigma: standard deviation used in the exponential

	Returns:
		gauss: numpy (3, 1) array of type float

	"""	

	#
	# You code goes here
	#
	
	#Let Gaussian function as f. Since our target pixel is located at position zero, 
	#calculate central value of gauss array by setting x to 0. 
	#other values are evaluated by x=-1 and x=1.
	f = lambda x:(np.exp(-(x**2)/(2*(sigma**2))))/(2*math.pi*(sigma**2))
	
	gauss = np.array([f(-1), f(0), f(1)])

	return gauss



def diff():
	"""Returns the derivative part corresponding to the central differences.
	The effect of this operator in x direction on function f should be:

			diff(x, y) = f(x + 1, y) - f(x - 1, y) 

	Returns:
		diff: (1, 3) array (float)
	"""

	#
	# You code goes here
	#

	return np.array([-1,0,1])

def create_sobel():
	"""Creates Sobel operator from two [3, 1] filters
	implemented in gaussian() and diff()

	Returns:
		sx: Sobel operator in x-direction
		sy: Sobel operator in y-direction
		sigma: Value of the sigma used to call gaussian()
		z: scaler of the operator
	"""

	#chosed these value of sigma and z because it returned most similar value compared to sobel filter given in docu. 
	sigma = 0.85
	z = 9
	
	#
	# You code goes here
	#

	#reshape corresponding gx and dy to calculate its convolution efficiently. 
	gx = reshape(gaussian(sigma))
	gy = gaussian(sigma)

	dx = diff()
	dy = reshape(dx)
	sx = z*gx*dx
	sy = z*dy*gy

	# do not change this
	return sx, sy, sigma, z

def apply_sobel(im, sx, sy):
	"""Applies Sobel filters to a greyscale image im and returns
	L2-norm.

	Args:
		im: (H, W) image (greyscale)
		sx, sy: Sobel operators in x- and y-direction

	Returns:
		norm: L2-norm of the filtered result in x- and y-directions
	"""

	im_norm = im.copy()


	#
	# Your code goes here
	#

	#Calculate the convolution of an image in both x and y direction.
	gx = signal.convolve2d(im, sx, mode = 'same')
	gy = signal.convolve2d(im, sy, mode = 'same')

	#Calculate the total normalized value and return it as im_norm
	w, h = im_norm.shape

	for i in range(w):
		for j in range(h):
			im_norm[i][j] = math.sqrt(gx[i][j]**2+gy[i][j]**2)

	return im_norm


def sobel_alpha(kx, ky, alpha):
	"""Creates a steerable filter for give kx and ky filters and angle alpha.
	The effect the created filter should be equivalent to 
		cos(alpha) I*kx + sin(alpha) I*ky, where * stands for convolution.

	Args:
		kx, ky: (3x3) filters
		alpha: steering angle

	Returns:
		ka: resulting kernel
	"""

	#
	# You code goes here
	#

	#Change angle from radian to dgree
	alpha = 180*(alpha/math.pi)

	#Evaluate the value(3*3 matrix) of ka
	ka = kx*math.cos(alpha)+ky*math.sin(alpha)

	return ka


"""
This is a multiple-choice question
"""

class EdgeDetection(object):

	# choice of the method
	METHODS = {
				1: "hysteresis",
				2: "non-maximum suppression"
	}

	# choice of the explanation
	# by "magnitude" we mean the magnitude of the spatial gradient
	# by "maxima" we mean the maxima of the spatial gradient
	EFFECT = {
				1: "it sharpens the edges by retaining only the local maxima",
				2: "it weakens edges with high magnitude if connected to edges with low magnitude",
				3: "it recovers edges with low magnitude if connected to edges with high magnitude",
				4: "it makes the edges thicker with Gaussian smoothing",
				5: "it aligns the edges with a dominant orientation"
	}

	def answer(self):
		"""Provide answer in the return value.
		This function returns tuples of two items: the first item
		is the method you will use and the second item is the explanation
		of its effect on the image. For example,
				((2, 1), (1, 1))
		means "hysteresis sharpens the edges by retaining only the local maxima",
		and "non-maximum suppression sharpens the edges by retaining only the local maxima"
		
		Any wrong answer will cancel the correct answer.
		"""

		return ((2, 1), (1,3))

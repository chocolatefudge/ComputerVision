import os
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image
import math



#
# Hint: you can make use of this function
# to create Gaussian kernels for different sigmas
#
def gaussian_kernel(fsize=3, sigma=1):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of guassian kernel

    Returns:
        Gaussian kernel
    '''
    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return G / G.sum()

def load_image(path):
    '''
    The input image is a)loaded, b) converted to greyscale, and c) converted to numpy array

    Args:
        path: the name of the inpit image
    Returns:
        img: numpy array containing image in greyscale
    '''
    img = plt.imread(path, format = 'jpeg')
    m,n,_ = img.shape
    result = np.empty((m,n))

    for i in range (m):
        for j in range (n):
            result[i][j] = 0.2126*img[i][j][0] + 0.7152*img[i][j][1] + 0.0772*img[i][j][2]

    return result

def smoothed_laplacian(image, sigmas, lap_kernel):
    '''
    The image is first smoothed by gaussian kernels for each sigma in the list of sigmas. Then laplacian operator is applied to each one.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''

    laplacian = laplacian_kernel()
    result = np.empty((len(sigmas), *image.shape))
    for i in range(len(sigmas)):
        img_copy = image.copy()
        gaussian = gaussian_kernel(7,sigmas[i])
        temp = convolve2d(img_copy, gaussian, mode = 'same')
        result[i] = convolve2d(temp, laplacian, mode = 'same')

    return result

def laplacian_of_gaussian(image, sigmas):
    '''
    Then laplacian of gaussian operator for every sigma in the list of sigmas is applied to the image.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''
    result = np.empty((len(sigmas), *image.shape))
    for i in range(len(sigmas)):
        img_copy = image.copy()
        filter = LoG_kernel(9, sigmas[i])
        result[i] = convolve2d(image, filter, mode = 'same')

    return result

def difference_of_gaussian(image, sigmas):
    '''
    Then difference of gaussian operator for every sigma in the list of sigmas is applied to the image.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''
    result = np.empty((len(sigmas), *image.shape))
    for i in range(len(sigmas)):
        img_copy = image.copy()
        filter = DoG(sigmas[i])
        result[i] = convolve2d(image, filter, mode = 'same')

    return result

def LoG_kernel(fsize=9, sigma=1):
    '''
    Define a LoG kernel.
    Tip: First calculate the second derivative of a gaussian and then discretize it.
    Args:
        fsize: kernel size
        sigma: sigma of guassian kernel

    Returns:
        LoG kernel
    '''
    gaussian = gaussian_kernel(fsize, sigma)
    laplacian = laplacian_kernel()
    result = convolve2d(gaussian, laplacian, mode = 'same')

    return result

def blob_detector(response):
    '''
    Find points with a response which is either maximum or minimum in their 3x3x3 neighborhood of scale space array.
    Tip: Ignore the first and the last row in every dimension for simplicity.
    Args:
        response: 3 dimensional response from LoG operator in scale space.

    Returns:
        list of 3-tuples (scale_index, row, column) containing the detected points.
    '''
    temp = []
    result = []
    tt, w, h = response.shape
    t = 1

    for i in range(1, w-1):
        for j in range(1, h-1):
            temp.append(response[t][i][j])
            temp.append(response[t][i][j-1])
            temp.append(response[t][i][j+1])
            temp.append(response[t][i-1][j])
            temp.append(response[t][i-1][j-1])
            temp.append(response[t][i-1][j+1])
            temp.append(response[t][i+1][j])
            temp.append(response[t][i+1][j-1])
            temp.append(response[t][i+1][j+1])
            temp.append(response[t-1][i][j])
            temp.append(response[t-1][i][j-1])
            temp.append(response[t-1][i][j+1])
            temp.append(response[t-1][i-1][j])
            temp.append(response[t-1][i-1][j-1])
            temp.append(response[t-1][i-1][j+1])
            temp.append(response[t-1][i+1][j])
            temp.append(response[t-1][i+1][j-1])
            temp.append(response[t-1][i+1][j+1])
            temp.append(response[t+1][i][j])
            temp.append(response[t+1][i][j-1])
            temp.append(response[t+1][i][j+1])
            temp.append(response[t+1][i-1][j])
            temp.append(response[t+1][i-1][j-1])
            temp.append(response[t+1][i-1][j+1])
            temp.append(response[t+1][i+1][j])
            temp.append(response[t+1][i+1][j-1])
            temp.append(response[t+1][i+1][j+1])

            if min(temp)==response[t][i][j] or max(temp)==response[t][i][j]:
                result.append((t, i, j))

            temp.clear()

    return result

def DoG(sigma):
    '''
    Define a DoG kernel. Please, use 7x7 kernels.
    Tip: First calculate the two gaussian kernels and return their difference. This is an approximation for LoG.

    Args:
        sigma: sigma of guassian kernel

    Returns:
        DoG kernel
    '''
    gaussian_1 = gaussian_kernel(7, sigma*math.sqrt(2))
    gaussian_2 = gaussian_kernel(7, sigma/math.sqrt(2))

    result = gaussian_1 - gaussian_2
    return result

def laplacian_kernel():
    '''
    Define a 3x3 laplacian kernel.
    Tip1: I_xx + I_yy
    Tip2: There are two possible correct answers.
    Args:
        none

    Returns:
        laplacian kernel
    '''
    return np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])


class Method(object):

    # select one or more options
    REASONING = {
        1: 'it is always more computationally efficient',
        2: 'it is always more precise.',
        3: 'it always has fewer singular points',
        4: 'it can be implemented with convolution',
        5: 'All of the above are incorrect.'
    }

    def answer(self):
        '''Provide answer in the return value.
        This function returns a tuple containing indices of the correct answer.
        '''

        return (1)

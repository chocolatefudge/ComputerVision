import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.


    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays
    '''
    # List for imgs and feats numpy arrays
    imgs = []
    feats = []

    # List for names of files
    img_files = []
    feat_files = []

    # paths
    img_path = path + '/facial_images'
    feats_path = path + '/facial_features'

    # For img files, make the list of file names in the directory facial_images
    # Using os.walk and os.path methods
    for root, dirs, files in os.walk(img_path, topdown=True):
        for name in files:
            img_files.append(os.path.join(root, name))

    # For img files, let's make python arrays
    for file in img_files:
        with open(file, 'rb') as infile:

            # Let's get width and height from header
            header = infile.readline()
            width, height, maxval = [int(item) for item in header.split()[1:]]

            # Make it into np array with data type uint8 cause the maxval is 255
            imgs.append(np.fromfile(infile, dtype=np.uint8).reshape((height, width)))


    # For feat files, make the list of file names in the directory facial_features
    # Using os.walk and os.path methods
    for root, dirs, files in os.walk(feats_path, topdown=True):
        for name in files:
            feat_files.append(os.path.join(root, name))

    # For feat files, let's make python arrays
    for file in feat_files:
        with open(file, 'rb') as infile:

            # Let's get width and height from header
            # The info of width and height is in second line
            header = infile.readline()
            header = infile.readline()
            width, height = [int(item) for item in header.split()]
            header = infile.readline()

            # Make it into np array with data type uint8 cause the maxval is 255
            feats.append(np.fromfile(infile, dtype=np.uint8).reshape((height, width)))

    return imgs, feats

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    # To make the y, x column and rows...
    m = (fsize-1.)/2.

    # Make y, x column and row which has values like -m -m+1 ... m
    # For example, '-2 -1 0 1 2' for size 5
    y,x = np.ogrid[-m:m+1,-m:m+1]

    # Using gaussian
    kernel = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    # Normalize the values so that the sum of all values be 1
    if kernel.sum() != 0:
        kernel /= kernel.sum()

    return kernel

def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    # Getting the height and width from x
    H, W = x.shape
    # Making the empty array sized H/2 * W/2
    downsample = np.empty((H//2, W//2))

    # Just put average of 4 values from 4 places to 1 place
    # Using np.mean method
    # It makes things downsampled by factor of 2
    for i in range(H//2):
        for j in range(W//2):
            # Using numpy mean function
            arr = [x[i*2,j*2],x[i*2+1,j*2],x[i*2,j*2+1],x[i*2+1, j*2+1]]
            downsample[i, j] = np.mean(arr)

    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = []

    # Appending first original img
    GP.append(img)

    # For nlevls-1 times
    for i in range(nlevels-1):
        # First, filter with the gaussian kernel
        img = convolve2d(img, gaussian_kernel(fsize, sigma), mode = 'same', boundary = 'symm')
        # And do the downsampling
        img = downsample_x2(img)
        GP.append(img)

    return GP

def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips:
        - Before doing this, let's take a look at the multiple choice questions that follow.
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''
    # SSD, Just subtract two vectors and squares it.
    distance = np.sum((v1-v2)**2)
    # Dot product, using np.sum and np.dot
    #distance = np.sum(np.dot(np.transpose(v1), v2))
    return distance


def sliding_window(img, feat, step=1):
    '''
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.

    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''

    min_score = None

    # Getting size of img
    img_h, img_w  = img.shape
    feat_h, feat_w  = feat.shape

    # Go around by pixels and check the distance
    for i in range(img_h):
        # Checking the out of range problem
        if i+feat_h >img_h:
            continue
        for j in range(img_w):
            # Checking the out of range problem
            if j+feat_w >img_w:
                continue
            # Crop the img to feature's size and calculate distance
            val = template_distance(img[i:i+feat_h, j:j+feat_w], feat)
            # Set the value if it's smallest for now
            if min_score == None or val<min_score:
                min_score = val

    return min_score


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''
        '''
        Since dot product just multiply every values, it may make things more sensitive to brightness.
        Also, the results seems better when it comes to ssd method.
        '''
        return (2, 2)  # TODO


def find_matching_with_scale(imgs, feats):
    '''
    Find face images and facial features that match the scales

    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []

    for img in imgs:
        # Initialize the values for the new img
        (score, g_im, feat) = (None, None, None)

        # For each pyramids and feature
        for pyramid in gaussian_pyramid(img, 3, 5, 1.4):

            # print('new_img', pyramid.shape)
            for feat_ in feats:
                # print("feat", feat_.shape)
                # Check the minimum score of this pyramid to feat
                val = sliding_window(pyramid,feat_)
                if val != None:
                # If it is minimum score for all pyramids for now,
                # Set the values to this condition
                    if score == None or val < score:
                        score = val
                        g_im = pyramid
                        feat = feat_


        # The last value is the condition for the minimum score
        if score != None:
            match.append((score, g_im, feat))
            # print(g_im.shape)

    return match

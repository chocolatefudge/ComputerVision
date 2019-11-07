import numpy as np
import scipy.signal as signal

def rgb2bayer(image):
    """Convert image to bayer pattern:
    [B G]
    [G R]

    Args:
        image: Input image as (H,W,3) numpy array

    Returns:
        bayer: numpy array (H,W,3) of the same type as image
        where each color channel retains only the respective 
        values given by the bayer pattern
    """
    assert image.ndim == 3 and image.shape[-1] == 3

    # otherwise, the function is in-place
    bayer = image.copy()

    #
    # You code goes here
    #

    #removing corresponding colors from each pixels. 

    for i in range(len(bayer)):
        for j in range(len(bayer[0])):
            if i%2 and j%2: #blue
                temp = bayer[i][j]
                bayer[i][j] = [0,0,temp[2]]
            elif i%2 or j%2: #green
                temp = bayer[i][j]
                bayer[i][j] = [0,temp[1],0]
            else: #red
                temp = bayer[i][j]
                bayer[i][j] = [temp[0],0,0]

    assert bayer.ndim == 3 and bayer.shape[-1] == 3
    return bayer

def nearest_up_x2(x):
    """Upsamples a 2D-array by a factor of 2 using nearest-neighbor interpolation.

    Args:
        x: 2D numpy array (H, W)

    Returns:
        y: 2D numpy array if size (2*H, 2*W)
    """
    assert x.ndim == 2
    h, w = x.shape

    #
    # You code goes here
    #

    #create an empty image
    y = np.empty((2*h, 2*w))

    #fill in the pixels with corresponding colors of the original image
    for i in range(h):
        for j in range(w):
            temp = x[i][j]
            y[2*i][2*j] = temp
            y[2*i+1][2*j] = temp
            y[2*i][2*j+1] = temp
            y[2*i+1][2*j+1] = temp

    assert y.ndim == 2 and \
            y.shape[0] == 2*x.shape[0] and \
            y.shape[1] == 2*x.shape[1]
    return y

def bayer2rgb(bayer):
    """Interpolates missing values in the bayer pattern.
    Note, green uses nearest neighbour upsampling; red and blue bilinear.

    Args:
        bayer: 2D array (H,W,C) of the bayer pattern
    
    Returns:
        image: 2D array (H,W,C) with missing values interpolated
        green_K: 2D array (3, 3) of the interpolation kernel used for green channel
        redblue_K: 2D array (3, 3) using for interpolating red and blue channels
    """
    assert bayer.ndim == 3 and bayer.shape[-1] == 3

    #
    # You code goes here
    #
    image = bayer.copy()
    # result = bayer.copy()
    # h,w = image[:,:,0].shape

    #idk part. 
    rb_k = np.array([[1/2, 1/2, 1/2], [1/2, 1, 1/2], [1/2, 1/2, 1/2]])
    g_k = np.array([[1/4,1/4,1/4], [1/4,1,1/4], [1/4,1/4,1/4]])


    image[:,:,0] = signal.convolve2d(image[:,:,0], rb_k, mode='same')
    image[:,:,2] = signal.convolve2d(image[:,:,2], rb_k, mode='same')
    image[:,:,1] = signal.convolve2d(image[:,:,1], g_k, mode='same')

    #orgval not zero, orgval smaller
    # for i in range(h):
    #     for j in range(w):
    #         img = image[i][j]
    #         res = result[i][j]
    #         if i%2 and j%2: #blue
    #             result[i][j] = [img[0], img[1], res[2]]
    #         elif i%2 or j%2: #green
    #             result[i][j] = [img[0], res[1], img[2]]
    #         else: #red
    #             result[i][j] = [res[0], img[1], img[2]]

    assert image.ndim == 3 and image.shape[-1] == 3 and \
                g_k.shape == (3, 3) and rb_k.shape == (3, 3)
    return image, g_k, rb_k

def scale_and_crop_x2(bayer):
    """Upscamples a 2D bayer pattern by factor 2 and takes the central crop.

    Args:
        bayer: 2D array (H, W) containing bayer pattern

    Returns:
        image_zoom: 2D array (H, W) corresponding to x2 zoomed and interpolated 
        one-channel image
    """
    assert bayer.ndim == 2

    #
    # You code goes here
    #

    #double up the image and than crop 
    h,w = bayer.shape
    cropped = nearest_up_x2(bayer.copy())
    cropped = cropped[h//2:h//2+h, w//2:w//2+w]

    assert cropped.ndim == 2
    return cropped

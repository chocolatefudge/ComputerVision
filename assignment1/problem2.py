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

    for i in range(len(bayer)):
        for j in range(len(bayer[0])):
            if i%2 and j%2:
                temp = bayer[i][j]
                bayer[i][j] = [0,0,temp[2]]
            elif i%2 or j%2:
                temp = bayer[i][j]
                bayer[i][j] = [0,temp[1],0]
            else:
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
    y = np.empty((2*h, 2*w))

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
    rb_k = np.array([[1/16, 1/16, 1/16], [1/16, 1/2, 1/16], [1/16, 1/16, 1/16]])
    g_k = np.array([[0,0,0], [0,1/2,1/2], [0,0,0]])


    image[:,:,0] = signal.convolve2d(image[:,:,0], rb_k, mode='same')
    image[:,:,2] = signal.convolve2d(image[:,:,2], rb_k, mode='same')
    image[:,:,1] = signal.convolve2d(image[:,:,1], g_k, mode='same')

    # x,y,z = image.shape
    # for i in range(x):
    #     for j in range(y):
    #         temp = image[i][j]
    #         image[i][j] = [temp[0]/2, temp[1]/2, temp[2]/2]

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
    h,w = bayer.shape
    cropped = nearest_up_x2(bayer.copy())
    cropped = cropped[h//2:h//2+h, w//2:w//2+w]

    assert cropped.ndim == 2
    return cropped

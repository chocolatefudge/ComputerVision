import numpy as np
from scipy.interpolate import griddata
import scipy.signal as signal

######################
# Basic Lucas-Kanade #
######################

def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.
    
    Args:
        im1: first image
        im2: second image
    
    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape
    
    Ix = np.empty_like(im1)
    Iy = np.empty_like(im1)
    It = np.empty_like(im1)

    #
    # Your code here
    #

    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sy = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

    Ix = signal.convolve2d(im1, sx, mode = "same")
    Iy = signal.convolve2d(im1, sy, mode = "same")
    It = im2-im1
    
    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It

def padding(x, y, m, n):
    if x<0:
        x = -x
    elif x>=m:
        x = 2*m-x-1
    
    if y<0:
        y = -y
    elif y>=n:
        y = 2*n-y-1

    return x,y

def calculate_uv(x,y,patch_size, Xss, Yss, XYs, XTs, YTs, m, n):

    t = patch_size//2

    X_ssum, Y_ssum, XY_sum, XT_sum, YT_sum = 0, 0, 0, 0, 0
    
    for i1 in range(-t, t+1):
        for j1 in range(-t, t+1):
            xc = x+i1
            yc = y+j1
            i, j = padding(xc,yc,m,n)
            X_ssum += Xss[i][j]
            Y_ssum += Yss[i][j]
            XY_sum += XYs[i][j]
            XT_sum += XTs[i][j]
            YT_sum += YTs[i][j]

    A = np.array([[X_ssum, XY_sum], [XY_sum, Y_ssum]])
    B = np.array([-XT_sum, -YT_sum])

    
    result = np.linalg.inv(A)@B.T 

    return result[0], result[1]

def pre_calc(Ix, Iy, It):
    Xss = np.multiply(Ix, Ix)
    Yss = np.multiply(Iy, Iy)
    XYs = np.multiply(Ix, Iy)
    XTs = np.multiply(Ix, It)
    YTs = np.multiply(Iy, It)

    return Xss, Yss, XYs, XTs, YTs

def compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2):
    """Computes one iteration of optical flow estimation.
    
    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: 0 or 1 specifying the region aggregation region
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction
    
    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    u = np.empty_like(Ix)
    v = np.empty_like(Iy)

    #
    # Your code here
    #

    m, n = Ix.shape
    Xss, Yss, XYs, XTs, YTs = pre_calc(Ix, Iy, It)

    for i in range (m):
        for j in range (n):
            uv, vv = calculate_uv(i,j,patch_size,Xss, Yss, XYs, XTs, YTs, m, n)
            u[i][j] = uv 
            v[i][j] = vv

    assert u.shape == Ix.shape and \
            v.shape == Ix.shape

    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.
    
    Args:
        im: input image
        u, v: optical flow in x and y direction
    
    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape
    
    im_warp = np.empty_like(im)
    #
    # Your code here
    #

    m, n = im.shape
    data1 = []
    data2 = []
    for i in range(m):
        for j in range(n):
            data1.append((i+u[i][j], j+v[i][j]))
            data2.append(im[i][j])
    
    grid_x, grid_y = np.mgrid[0:m, 0:n]
    im_warp = griddata(np.asarray(data1), np.asarray(data2), (grid_x, grid_y), method = 'nearest')

    assert im_warp.shape == im.shape

    return im_warp

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    d = 0.0
    #
    # Your code here
    #

    Ix, Iy, It = compute_derivatives(im1, im2)
    u, v = compute_motion(Ix, Iy, It)

    d = np.sum(np.square(np.multiply(u, Ix) + np.multiply(v, Iy) + It))

    
    assert isinstance(d, float)
    return d

####################
# Gaussian Pyramid #
####################

#
# this function implementation is intentionally provided
#
def gaussian_kernel(fsize, sigma):
    """
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: deviation of the Guassian

    Returns:
        kernel: (fsize, fsize) Gaussian (normalised) kernel
    """

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    return G / G.sum()

def downsample_x2(x, fsize=5, sigma=1.4):
    """
    Downsampling an image by a factor of 2
    Hint: Don't forget to smooth the image beforhand (in this function).

    Args:
        x: image as numpy array (H x W)
        fsize and sigma: parameters for Guassian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as numpy array (H/2 x W/2)
    """


    #
    # Your code here
    #

    return x

def gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4):
    '''
    A Gaussian pyramid is a sequence of downscaled images
    (here, by a factor of 2 w.r.t. the previous image in the pyramid)

    Args:
        img: face image as numpy array (H * W)
        nlevels: num of level Gaussian pyramid, in this assignment we will use 3 levels
        fsize: gaussian kernel size, in this assignment we will define 5
        sigma: sigma of guassian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of gaussian downsampled images in ascending order of resolution
    '''
    
    #
    # Your code here
    #

    return [img]

###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(im1, im2, pyramid1, pyramid2, n_iter=3):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.
    
    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations
    
    Returns:
        u: OF in x direction
        v: OF in y direction
    """
    assert im1.shape == im2.shape
    
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)

    #
    # Your code here
    #
    
    assert u.shape == im1.shape and \
            v.shape == im1.shape
    return u, v


###############################
#   Multiple-choice question  #
###############################
def task9_answer():
    """
    Which statements about optical flow estimation are true?
    Provide the corresponding indices in a tuple.

    1. For rectified image pairs, we can estimate optical flow 
       using disparity computation methods.
    2. Lucas-Kanade method allows to solve for large motions in a principled way
       (i.e. without compromise) by simply using a larger patch size.
    3. Coarse-to-fine Lucas-Kanade is robust (i.e. negligible difference in the 
       cost function) to the patch size, in contrast to the single-scale approach.
    4. Lucas-Kanade method implicitly incorporates smoothness constraints, i.e.
       that the flow vector of neighbouring pixels should be similar.
    5. Lucas-Kanade is not robust to brightness changes.

    """

    return (-1, -2, -3)
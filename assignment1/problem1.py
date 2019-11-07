import numpy as np
import matplotlib.pyplot as plt

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    
    #
    # You code here
    #
    plt.imshow(img, interpolation='nearest')
    plt.show()
    return

def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """
    
    #
    # You code here
    #
    np.save(path,img)
    return


def load_npy(path):
    """ Load and return the .npy file:

    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    return np.load(path)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """
    
    #
    # You code here
    #
    return np.flip(img,0)

def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    #
    # You code here
    #
    plt.imshow(np.vstack([img1, img2]), interpolation='nearest')
    plt.show()
    return

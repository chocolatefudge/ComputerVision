import numpy as np
import os
from PIL import Image

#
# Task 1
#
def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and
    d is the dimensionality (height*width for greyscale).
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: tuple with two elements (height, width)
    """
    
    #
    # You code here
    #
    
    #Empty list for image file names. 
    img_files = []

    #Empty list for image files
    imgs = []

    #List all the files in target path using os.walk(path)
    #Save file names to img_files. 
    for root, dirs, files in os.walk(path):
        for name in files:
            img_files.append(os.path.join(root, name))

    #Open each file in img_files and read header first to get h, w, max value info. 
    #Then, read image file's content and save it to imgs array. 
    for file in img_files:
        with open(file, 'rb') as infile:
            header = infile.readline()
            width, height, maxval = [int(item) for item in header.split()[1:]]
            #reshaping needed to organize whole pixels from one image in one row
            imgs.append(np.fromfile(infile, dtype=np.uint8).reshape((height*width)))
            
    #Change list to numpy array and return
    result = np.asarray(imgs)
    return result, (height, width)

#
# Task 2
#

"""
This is a multiple-choice question
"""

class PCA(object):

    # choice of the method
    METHODS = {
                1: "SVD",
                2: "Eigendecomposition"
    }

    # choice of reasoning
    REASONING = {
                1: "it can be applied to any matrix and is more numerically stable",
                2: "it is more computationally efficient for our problem",
                3: "it allows to compute eigenvectors and eigenvalues of any matrix",
                4: "we can find the eigenvalues we need for our problem from the singular values",
                5: "we can find the singular values we need for our problem from the eigenvalues"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of PCA
            - the following integers provide the reasoning for your choice

        For example (made up):
            (2, 1, 5) means
            "I will use eigendecomposition because
                - we can apply it to any matrix
                - we need singular values which we can obtain from the eigenvalues"
        """

        return (1, 1, 2)

#
# Task 3
#

def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, N) bases with principal components
        lmb: (N, ) corresponding variance
    """
    
    #Each rows are image vectors. 
    #N: Number of images, M:Dimension
    N, M = X.shape
    
    #Transpose X to arrange image vectors in column space. 
    X_tr = np.transpose(X)
    #now each columns are faces(imges)

    #Calculate mean from X_tr(along the column axis)
    x_mean = np.mean(X_tr, axis = 1)

    #Subract mean from each corresponding elements of X_tr
    for i in range(M):
        for j in range(N):
            X_tr[i][j]-=x_mean[i]

    #SVD for X_tr - X_mean. 
    u, s, vt = np.linalg.svd(X_tr)

    #For getting variances, lambda = s^2/N is used for each eigenvalue in s. 
    for i in range(s.shape[0]):
        s[i] = s[i]*s[i]/N

    return u, s

#
# Task 4
#

def basis(u, s, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) contains principal components.
        For example, i-th vector is u[:, i]
        s: (M, ) variance along the principal components.
    
    Returns:
        v: (M, D) contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """


    #calculate sum of the total singular values
    #get a percentile
    sum_total = np.sum(s)*p

    #sum until total value reaches the percentile
    sum_temp = 0
    idx = 0

    while sum_temp<sum_total:
        sum_temp+=s[idx]
        idx+=1

    #return the part of the array(there are only D principal components)
    result = u[:,:idx]

    return result

#
# Task 5
#
def project(face_image, u):
    """Project face image to a number of principal
    components specified by num_components.
    
    Args:
        face_image: (N, ) vector (N=h*w) of the face
        u: (N,M) matrix containing M principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (N, ) vector, projection of face_image on 
        principal components
    """
    
    #Projection of a vector to matrix. 

    result = u@np.linalg.inv(np.transpose(u)@u)@np.transpose(u)@face_image

    return result

#
# Task 6
#

"""
This is a multiple-choice question
"""
class NumberOfComponents(object):

    # choice of the method
    OBSERVATION = {
                1: "The more principal components we use, the sharper is the image",
                2: "The fewer principal components we use, the smaller is the re-projection error",
                3: "The first principal components mostly correspond to local features, e.g. nose, mouth, eyes",
                4: "The first principal components predominantly contain global structure, e.g. complete face",
                5: "The variations in the last principal components are perceptually insignificant; these bases can be neglected in the projection"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple describing you observations

        For example: (1, 3)
        """

        return 0 #Todo


#
# Task 7
#
def search(Y, x, u, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) centered array with N d-dimensional features
        x: (1, M) image we would like to retrieve
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M)
    """
    #Edge Case: return x itself. 
    N, M = Y.shape
    if top_n == 1:
        return x

    #Transpose both matrix to fit in shape
    new_u = np.transpose(u)
    new_Y = np.transpose(Y)

    #Evaluate projection coefficients a's by multiplying two matrix. 
    proj_matrix = new_u@new_Y
    proj_vector = new_u@x.reshape((x.shape[0], 1))

    #Calculate distance between each vector in proj_matrix with proj_vector and save to distance list. 
    D, N = proj_matrix.shape
    distance = []
    for i in range(N):
        dist = np.linalg.norm(proj_vector-proj_matrix[:,i])
        distance.append(dist)

    #Sort distance values and filter top_n values. 
    dis_sort = distance.copy()
    top_idx = np.argsort(dis_sort)[-top_n:]

    # print(top_values)
    # #Find top value's index and save to top_idx
    # top_idx = []
    # for j in top_values:
    #     top_idx.append(distance.index(j))

    #Get corresponding original image vector from Y. (Each row is vector)
    result = np.empty([top_n, M])
    for k in range(len(top_idx)):
        result[k] = Y[top_idx[k]]

    return result

#
# Task 8
#
def interpolate(x1, x2, u, N):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (1, M) array, the first image
        x2: (1, M) array, the second image
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        N: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (N, M) interpolated results. The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """
    
    #Project x1 and x2 to u. 
    x1_proj = project(x1, u)
    x2_proj = project(x2, u)

    #For each componant in x1_proj and x2_proj, get linspace and append it to result. 
    M = x1_proj.shape[0]
    result = np.empty([N, M])
    for i in range(M):
        result[:,i] = np.linspace(x1_proj[i], x2_proj[i], num=N)

    return result

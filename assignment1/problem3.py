import numpy as np

def load_points(path):
    '''
    Load points from path pointing to a numpy binary file (.npy).
    Image points are saved in 'image'
    Object points are saved in 'world'

    Returns:
        image: A Nx2 array of 2D points form image coordinate
        world: A N*3 array of 3D points form world coordinate
    '''


    #idk part how to load the points from npz file
    data = np.load('data/points.npz') # loading data
    lst = data.files # It includes the key image and world

    image_pts = data[lst[0]]#np.empty((100, 3))
    world_pts = data[lst[1]]#np.empty((100, 4))

    # sanity checks
    assert image_pts.shape[0] == world_pts.shape[0]
    # As I checked, they're both 75

    # homogeneous coordinates
    assert image_pts.shape[1] == 3 and world_pts.shape[1] == 4
    return image_pts, world_pts


def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product

    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)

    Returns:
        A: (2*N, 12) matrix A
    """

    N, _ = x.shape
    assert N == X.shape[0] #N is 75
    print(N)

    A = np.empty((150, 12))

    for i in range(N):
        x1 = x[i][0]
        y1 = x[i][1]
        temp = X[i]

        #print(A[0:4])

        A[2*i,0:4] = 0
        A[2*i,4:8] = -np.transpose(temp)
        A[2*i,8:12] = y1*np.transpose(temp)

        A[2*i+1,0:4] = np.transpose(temp)
        A[2*i+1,4:8] = 0
        A[2*i+1,8:12] = -x1*np.transpose(temp)


    assert A.shape[0] == 2*N and A.shape[1] == 12
    return A


def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.

    Args:
        A: (2*N, 12) matrix A

    Returns:
        P: (3, 4) projection matrix P
    """
    u,s,v = np.linalg.svd(A)
    idx = np.argmin(s)

    P = np.transpose(v)[idx]
    #print(P)
    mat_p = np.empty((3,4))
    #print(P[0:4])
    mat_p[0] = P[0:4]
    mat_p[1] = P[4:8]
    mat_p[2] = P[8:12]
    print(mat_p)

    return mat_p


def solve_KR(P):
    """Using th RQ-decomposition find K and R
    from the projection matrix P.
    Hint 1: you might find scipy.linalg useful here.
    Hint 2: recall that K has 1 in the the bottom right corner.
    Hint 3: RQ decomposition is not unique (up to a column sign).
    Ensure positive element in K by inverting the sign in K columns
    and doing so correspondingly in R.

    Args:
        P: 3x4 projection matrix.

    Returns:
        K: 3x3 matrix with intrinsics
        R: 3x3 rotation matrix
    """

    M = np.hsplit(P, np.array([3,6]))[0]
    print(M)
    R, K = np.linalg.qr(M)

    val = K[2][2]
    R = val*R
    K = K/val

    return K, R

def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.

    Args:
        P: 3x4 projection matrix

    Returns:
        c: 3x1 camera center coordinate in the world frame
    """
    u,s,v = np.linalg.svd(P)
    idx = np.argmin(s)
    #print(s[idx])
    #assert s[idx]==0

    c = np.transpose(v)[idx]
    c = c*(1/c[3])

    return c[0:2]

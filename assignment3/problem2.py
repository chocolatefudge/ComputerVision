import numpy as np
import matplotlib.pyplot as plt
import random
import math


def load_pts_features(path):
    """ Load interest points and SIFT features.

    Args:
        path: path to the file pts_feats.npz

    Returns:
        pts: coordinate points for two images;
             an array (2,) of numpy arrays (N1, 2), (N2, 2)
        feats: SIFT descriptors for two images;
               an array (2,) of numpy arrays (N1, 128), (N2, 128)
    """

    #idk part how to load the points from npz file
    data = np.load('data/pts_feats.npz') # loading data
    lst = data.files # It includes the key pts and feats

    # Using keys, get the data
    pts = data[lst[0]]
    feats = data[lst[1]]

    return pts, feats

def min_num_pairs():
    # As discussed in lecture, it is 4.
    return 4

def pickup_samples(pts1, pts2):
    """ Randomly select k corresponding point pairs.
    Note that here we assume that pts1 and pts2 have
    been already aligned: pts1[k] corresponds to pts2[k].

    This function makes use of min_num_pairs()

    Args:
        pts1 and pts2: point coordinates from Image 1 and Image 2

    Returns:
        pts1_sub and pts2_sub: N_min randomly selected points
                               from pts1 and pts2
    """

    # Get the random 4 indexes in range
    random_num = random.sample(range(0, len(pts2)), min_num_pairs())
    pts1_sub = []
    pts2_sub = []
    # For each random numbers, take the pts points into the arrays
    for i in random_num:
        pts1_sub.append(pts1[i])
        pts2_sub.append(pts2[i])

    return pts1_sub, pts2_sub


def compute_homography(pts1, pts2):
    """ Construct homography matrix and solve it by SVD

    Args:
        pts1: the coordinates of interest points in img1, array (N, 2)
        pts2: the coordinates of interest points in img2, array (M, 2)

    Returns:
        H: homography matrix as array (3, 3)
    """

    # Making A matrix using 4 interest points, for every interest points,
    # we need 2 rows with 9 columns. So A becomes 8 by 9 matrix.
    A = np.empty((8,9))

    pts1_sub, pts2_sub = pickup_samples(pts1, pts2)
    # For every 4 points
    for i in range(4):

        # Extracting each points from pts1 and pts2
        x = pts1_sub[i][0]
        y = pts1_sub[i][1]
        x_ = pts2_sub[i][0]
        y_ = pts2_sub[i][1]

        # Putting the values to A matrix to get the H
        A[i*2][0] = 0
        A[i*2][1] = 0
        A[i*2][2] = 0
        A[i*2][3] = x
        A[i*2][4] = y
        A[i*2][5] = 1
        A[i*2][6] = (-1)*x*y_
        A[i*2][7] = (-1)*y*y_
        A[i*2][8] = (-1)*y_

        A[i*2+1][0] = (-1)*x
        A[i*2+1][1] = (-1)*y
        A[i*2+1][2] = -1
        A[i*2+1][3] = 0
        A[i*2+1][4] = 0
        A[i*2+1][5] = 0
        A[i*2+1][6] = x*x_
        A[i*2+1][7] = y*x_
        A[i*2+1][8] = x_

    # Doing the SVD decomposition
    u,s,v = np.linalg.svd(A)
    # Picking right most vector
    H = v[8]
    # reshape it to 3*3 size
    H = np.reshape(H, (3,3))
    return H


def transform_pts(pts, H):
    """ Transform pst1 through the homography matrix to compare pts2 to find inliners

    Args:
        pts: interest points in img1, array (N, 2)
        H: homography matrix as array (3, 3)

    Returns:
        transformed points, array (N, 2)
    """


    # Making homogeneous array of pts by adding 1 to last colums
    ones = np.ones((len(pts), 1), dtype=np.int8)
    homo_pts = np.append(pts, ones, axis=1)

    # Transform pts by multiplying H and homo_pts
    trans_pts = np.dot(homo_pts, np.transpose(H))

    # Make matrix from N * 3 to N * 2 by dividing x and y by trans_pts[i][2]
    trans_pts_ = np.empty((len(pts), 2))

    for i in range(len(pts)):
        trans_pts_[i][0] = trans_pts[i][0]/trans_pts[i][2]
        trans_pts_[i][1] = trans_pts[i][1]/trans_pts[i][2]

    return trans_pts_


def count_inliers(H, pts1, pts2, threshold=5):
    """ Count inliers
        Tips: We provide the default threshold value, but you’re free to test other values
    Args:
        H: homography matrix as array (3, 3)
        pts1: interest points in img1, array (N, 2)
        pts2: interest points in img2, array (N, 2)
        threshold: scale down threshold

    Returns:
        number of inliers
    """
    # inliers count
    count = 0

    # First, transform pts1 using H matrix
    pts1_ = transform_pts(pts1, H)

    # Calculate distances between transformed pts1 and pts2
    for i in range(len(pts2)):
        distance = math.sqrt((pts1_[i][0]-pts2[i][0])**2 + (pts1_[i][1]-pts2[i][1])**2)
        if distance<threshold: #if distance is smaller than threshold, add 1
            count += 1
    return count


def ransac_iters(w=0.5, d=min_num_pairs(), z=0.99):
    """ Computes the required number of iterations for RANSAC.

    Args:
        w: probability that any given correspondence is valid
        d: minimum number of pairs
        z: total probability of success after all iterations

    Returns:
        minimum number of required iterations
    """
    # Let's calculate k which is required num of iterations for ransac
    # As discussed in lecture, k can be expressed as below.
    k = round(math.log(1-z)/math.log(1-w**d))
    return k


def ransac(pts1, pts2):
    """ RANSAC algorithm

    Args:
        pts1: matched points in img1, array (N, 2)
        pts2: matched points in img2, array (N, 2)

    Returns:
        best homography observed during RANSAC, array (3, 3)
    """

    # To keep track of inlier counts, use max_inlier variable
    max_inlier = None
    # To keep track of homography, use best_H variable
    best_H = None

    # For ransac_iters times, doing ransac algorithm
    for i in range(ransac_iters()):
        H = compute_homography(pts1, pts2) # Compute homography
        cnt = count_inliers(H, pts1, pts2)
        # If it's first loop, put value directly to max_inlier
        if max_inlier == None:
            max_inlier = cnt
            best_H = H
        # If it's maximum inlier, then save homography
        if cnt > max_inlier:
            max_inlier = cnt
            best_H = H
    # After the loop, best_H has the best homography

    return best_H


def find_matches(feats1, feats2, rT=0.8):
    """ Find pairs of corresponding interest points with distance comparsion
        Tips: We provide the default ratio value, but you’re free to test other values

    Args:
        feats1: SIFT descriptors of interest points in img1, array (N, 128)
        feats2: SIFT descriptors of interest points in img1, array (M, 128)
        rT: Ratio of similar distances

    Returns:
        idx1: list of indices of matching points in img1
        idx2: list of indices of matching points in img2
    """
    # List to store the indexes
    idx1 = []
    idx2 = []

    # For feats1.shape[0] times and feats2.shape[0] times
    for i in range(feats1.shape[0]):
        min_dist = None
        sec_dist = None
        min_index = None
        sec_index = None
        for j in range(feats2.shape[0]):
            # Get the distance
            dist = np.linalg.norm(feats1[i]-feats2[j])

             # If it's first loop
            if min_dist == None:
                min_dist = dist
                min_index = j
            # If dist is min_dist until now
            elif dist < min_dist:
                sec_dist = min_dist
                min_dist = dist
                sec_index = min_index
                min_index = j
            # Else
            elif dist > min_dist:
                # If sec_dist has nothing
                if sec_dist == None:
                    sec_dist = dist
                    sec_index = j
                # If dist is bigger than minimum but smaller than second min
                elif dist < sec_dist:
                    sec_dist = dist
                    sec_index = j
        # After all the loops, check if min/sec is smaller than rT and add it to list.
        if sec_dist!=None and min_dist/sec_dist < rT:
            idx1.append(i)
            idx2.append(min_index)

    return idx1, idx2


def final_homography(pts1, pts2, feats1, feats2):
    """ re-estimate the homography based on all inliers

    Args:
       pts1: the coordinates of interest points in img1, array (N, 2)
       pts2: the coordinates of interest points in img2, array (M, 2)
       feats1: SIFT descriptors of interest points in img1, array (N, 128)
       feats2: SIFT descriptors of interest points in img1, array (M, 128)

    Returns:
        ransac_return: refitted homography matrix from ransac fucation, array (3, 3)
        idxs1: list of matched points in image 1
        idxs2: list of matched points in image 2
    """
    # find corresponding pairs by find_matches and get the index of them
    idx1, idx2 = find_matches(feats1, feats2)
    # numpy array which has 2 columns
    idxs1, idxs2 = np.empty((len(idx1),2)), np.empty((len(idx1),2))

    # By index, get the value
    for i in range(len(idx1)):
        idxs1[i] = pts1[idx1[i]]
        idxs2[i] = pts2[idx2[i]]

    #Using values, do ransac and get the best homography
    ransac_return = ransac(idxs1, idxs2)

    return ransac_return, idx1, idx2

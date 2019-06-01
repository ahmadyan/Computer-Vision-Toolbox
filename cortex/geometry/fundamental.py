import numpy as np
import cv2
from geometry.multiview import homg


def estimate_fundamental_matrix(points1, points2, method=cv2.FM_RANSAC):
    '''
    Computes the fundamental matrix from corresponding points x1, x2 using the 8 point algorithm.
    :return: fundamental matrix F
    '''

    F, mask = cv2.findFundamentalMat(points1, points2, method)
    #F, mask = cv2.findFundamentalMat(points1, points2, cv2.RANSAC, 3.0, 0.999)

    if F is None or F.shape == (1, 1):
        raise Exception('No fundamental matrix found')
    elif F.shape[0] > 3:
        # more than one matrix found, pick the first
        F = F[0:3, 0:3]

    return np.matrix(F)


def SymmetricEpipolarDistance(F, x1, x2):
    '''
        Calculates the sum of the distances from the points to the epipolar lines.
        MVG, p. 288, Eq. 11.10

    :param F: Mat fundamental matrix
    :param x1: Vec2 point in the first image
    :param x2: Vec2 point in the second image
    :return: scalar: symmetric epipolar distance
    '''
    x = homg(x1)
    y = homg(x2)
    Fx = np.matmul(F, x).T            #Vec3
    Ft_y = np.matmul(F.T, y)        #Vec3
    yFx = y.dot(Fx)                 #double

    e = yFx*yFx * ((1/np.sum(np.square(Fx))) + (1/np.sum(np.square(Ft_y))))
    return e


def DecomposeFundamentalMatrix(F, K):
    return K * F * np.linalg.inv(K)


def compute_essential(F, K):
    """ Compute the Essential matrix, and R1, R2 """
    return (K.T).dot(F).dot(K)


def compute_epipole(F):
    """ Computes the (right) epipole from a
        fundamental matrix F.
        (Use with F.T for left epipole.) """

    # return null space of F (Fx=0)
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]

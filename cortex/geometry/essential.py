import cv2
import numpy as np
from numpy.linalg import det, norm
from scipy.linalg import svd


def decompose_E(E):
    """
    Decomposes the essential matrix and returns two possible rotations
    R1,R2, and the relative translation. Only consider cases where
    points are in front of the camera.
    TODO Perform Cheirality check:
    See https://github.com/opencv/opencv/blob/710506e9e220880423ebda5cc5d6d8d72df29a29/modules/calib3d/src/five-point.cpp#L506
    [R1,t], [R2,t], [R1,-t], [R2,-t]
    """
    U, S, Vt = svd(E)
    W = np.float32([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
    R = U.dot(W).dot(Vt)
    t = U[:, 2] / np.linalg.norm(U[:, 2])
    assert(np.fabs(np.linalg.norm(t) - 1.0) < 1e-6)

    R1 = U.dot(W).dot(Vt)
    if det(R1) < 0:
        R1 = -R1
    R2 = U.dot(W.T).dot(Vt)
    if det(R2) < 0:
        R2 = -R2

    return R1, R2, t

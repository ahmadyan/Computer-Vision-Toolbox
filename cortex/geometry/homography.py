import numpy as np
import cv2
import geometry.se3 as lie
from geometry.multiview import homg


def estimate_relative_pose_using_homography(points1, points2, K=None):
    # Compute Homography H
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Decompose Homography using Matis' method
    # todo: implement Zhang's method
    _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    return H


def SymmetricGeometricDistance(H, x1, x2):
    '''
        Calculate symmetric geometric cost:
        D(H * x1, x2)^2 + D(H^-1 * x2, x1)
    :param H: Mat3
    :param x1: Vec2
    :param x2: Vec2
    :return:
    '''

    Hx1 = np.matmul(H, homg(x1))
    Hinv_x2 = np.matmul(np.linalg.inv(H), homg(x2))
    Hx1 /= Hx1[2]
    Hinv_x2 /= Hinv_x2[2]
    e = np.sum(np.square((Hx1[:2] - x2))) + np.sum(np.square((Hinv_x2[:2] - x1)))
    return e


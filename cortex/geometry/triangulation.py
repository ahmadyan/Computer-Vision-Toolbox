import numpy as np
import cv2
from cortex.geometry.multiview import homg
from cortex.geometry.multiview import project


# Expects its input in pixel space
def triangulate3d(x1, x2, p1, p2, K):
    print(x1, x2, p1, p2, K)
    transformation1 = p1.camera(K)
    transformation2 = p2.camera(K)
    point_4d_hom = cv2.triangulatePoints(transformation1, transformation2, np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1))
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :].T
    return point_3d


# OpenCV (similar to OpenMVG) does have a triangulateNViews method in opencv_contrib/modules/sfm however, it does not
# have a Python wrapper. So we have to re-implement it here.
def triangulateNviews(x, P):
    '''
    This is not optimal, since it is not minimizing reprojection error
    Reference: Kier Mierle's thesis, p. 102
    :param Xs: Matrix of 2xN points, in metric space
    :param Ps: List of camera poses (SE3)
    :param K: Shared intrinsic matrix
    :return: 3d point (1x3)
    '''

    nviews = len(P)
    A = np.zeros((3*nviews, 4 + nviews))
    for i in range(nviews):
        A[3*i : 3*i + 3, 0:4] = -P[i].matrix3x4()
        A[3*i : 3*i + 2, 4+i] = x[:, i]
        A[3*i + 2, 4 + i] = 1.0

    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
    ncol = A.shape[1]
    # The last column of matrix V is the solution
    # the first four elements are the homgenous 3D points, the rest are scaling factors
    # Numpy returns the transposed matrix V, so we transpose it back
    Xh = vh.T[0:4, ncol-1]
    alpha = vh.T[4:, ncol-1]
    X3D = project(Xh)

    error = 0.
    for i in range(nviews):
        # we have x_i = P_i * X / alpha_i
        error += np.linalg.norm(project(P[i] * X3D, 3) - (alpha[i] * x[:,i]))
    return X3D, error


def triangulateNViewsAlgebraic(x, P):
    n = len(P)
    A = np.zeros((4,4))
    for i in range(n):
        T = P[i].matrix3x4()
        y = homg(x[:, i])
        norm = np.linalg.norm(y)
        if norm > 0:
            y = y / norm

        ydot = y * y.T
        cost = T - np.dot(ydot.reshape(1, 3), T)
        A += np.dot(cost.T, cost)


    w, v = np.linalg.eig(A)
    x3d = v[0, :] / v[0, 3]
    return x3d

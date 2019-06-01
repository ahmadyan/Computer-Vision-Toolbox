import numpy as np
import math

def perspective_transformation_matrix(src, dst):
    """ Returns perspective transformation that will map src points to dst.
        Similar to cv2.getPerspectiveTransform
    """
    m = []
    for (x, y), (X, Y) in zip(src, dst):
        m.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    A = np.matrix(m, dtype=np.float)
    B = np.array(dst).reshape(8)
    af = np.linalg.solve(A, B)
    return np.append(np.array(af).reshape(8), 1).reshape((3, 3))


def affine_transformation_matrix(translate, scale, shear, angle):
    translation_matrix = np.array([[1, 0, 0], [0, 1, 0], [translate[0], translate[1], 1]])
    scale_matrix = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]])
    shear_matrix = np.array([[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]])
    rotation_matrix = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    m = translation_matrix * scale_matrix * shear_matrix * rotation_matrix

    return m

def identity_transformation():
    return np.hstack((np.eye(3, 3), np.zeros((3, 1))))

import numpy as np
import math
from pyquaternion import Quaternion


def is_matrix_orthogonal(R):
    return np.linalg.norm(np.linalg.inv(R) - R.T)


def reorthogonalize(r, threshold = 1e-16):
    '''
    Reorthogonalizes the rotation matrix R, such that inv(R) = R'
    Based on Direction Cosine Matrix IMU: Theory by William Premerlani and Paul Bizard Eq 19-21
    :param r: rotation matrix
    :param threshold: threshold for the error
    :return: orthogonal rotation matrix
    '''
    x = r[0,:]
    y = r[1,:]
    z = r[2,:]
    error = np.dot(x,y)

    # rotation matrix is messed up
    if abs(error) > threshold:
        # spread the error over x and y
        x2 = x - (error/2) * y
        y2 = y - (error/2) * x
        z2 = np.cross(x2, y2)
        # renormalize the matrix using Taylor expansion
        r[0,:] = 1 / 2 * (3 - np.dot(x2, x2)) * x2
        r[1,:] = 1 / 2 * (3 - np.dot(y2, y2)) * y2
        r[2,:] = 1 / 2 * (3 - np.dot(z2, z2)) * z2

    return r


def rt2qt(rotation, translation):
    quaternion = Quaternion(matrix=rotation).elements
    return [quaternion[0], quaternion[1], quaternion[2], quaternion[3], translation[0], translation[1], translation[2]]


def qt2rt(quaternion, translation):
    T = quaternion.transformation_matrix
    T[0:3, 3] = translation
    return T


def SO3_log(C):
    ''' Compute the matrix log of the rotation matrix C '''
    epsilon = 0.0001
    epsilon2 = 0.001
    if ((abs(C[0, 1] - C[1, 0]) < epsilon) and (abs(C[0, 2] - C[2, 0]) < epsilon) and (
            abs(C[1, 2] - C[2, 1]) < epsilon)):
        # singularity found
        # first check for identity matrix which must have +1 for all terms
        # in leading diagonaland zero in other terms
        if ((abs(C[0, 1] + C[1, 0]) < epsilon2) and (abs(C[0, 2] + C[2, 0]) < epsilon2) and (
                abs(C[1, 2] + C[2, 1]) < epsilon2) and (
                abs(C[0, 0] + C[1, 1] + C[2, 2] - 3) < epsilon2)):  # this singularity is identity matrix so angle = 0
            return np.zeros(3)  # zero angle, arbitrary axis
        # otherwise this singularity is angle = 180
        angle = np.pi
        xx = (C[0, 0] + 1) / 2.
        yy = (C[1, 1] + 1) / 2.
        zz = (C[2, 2] + 1) / 2.
        xy = (C[0, 1] + C[1, 0]) / 4.
        xz = (C[0, 2] + C[2, 0]) / 4.
        yz = (C[1, 2] + C[2, 1]) / 4.
        if ((xx > yy) and (xx > zz)):  # C[0][0] is the largest diagonal term
            if (xx < epsilon):
                x = 0
                y = np.sqrt(2) / 2.
                z = np.sqrt(2) / 2.
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif (yy > zz):  # C[1][1] is the largest diagonal term
            if (yy < epsilon):
                x = np.sqrt(2) / 2.
                y = 0
                z = np.sqrt(2) / 2.
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # C[2][2] is the largest diagonal term so base result on this
            if (zz < epsilon):
                x = np.sqrt(2) / 2.
                y = np.sqrt(2) / 2.
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return angle * np.array((x, y, z))
    s = np.sqrt(
        (C[2, 1] - C[1, 2]) * (C[2, 1] - C[1, 2]) + (C[0, 2] - C[2, 0]) * (C[0, 2] - C[2, 0]) + (C[1, 0] - C[0, 1]) * (
                    C[1, 0] - C[0, 1]))  # used to normalise
    if (abs(s) < 0.001):
        # prevent divide by zero, should not happen if matrix is orthogonal and should be
        # caught by singularity test above, but I've left it in just in case
        s = 1

    angle = np.arccos((C[0, 0] + C[1, 1] + C[2, 2] - 1) / 2.)
    x = (C[2, 1] - C[1, 2]) / s
    y = (C[0, 2] - C[2, 0]) / s
    z = (C[1, 0] - C[0, 1]) / s
    return angle * np.array((x, y, z))


def SO3_exp(phi):
    tiny = 1e-12
    # check for small angle
    nr = np.linalg.norm(phi)
    if nr < tiny:
        # ~ # If the angle (nr) is small, fall back on the series representation.
        # C = VecToRotSeries(phi,10)
        C = np.eye(3)
    else:
        R = Hat(phi)
        C = np.eye(3) + np.sin(nr) / nr * R + (1 - np.cos(nr)) / (nr * nr) * np.dot(R, R)
    return C


def aa2r(axis, angle):
    """Generate the rotation matrix from the axis-angle notation.

        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    """
    axis = np.matrix(axis)
    ca = math.cos(angle)
    sa = math.sin(angle)
    C = 1 - ca

    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    r = np.zeros((3,3))
    # Update the rotation matrix.
    r[0, 0] = x*xC + ca
    r[0, 1] = xyC - zs
    r[0, 2] = zxC + ys
    r[1, 0] = xyC + zs
    r[1, 1] = y*yC + ca
    r[1, 2] = yzC - xs
    r[2, 0] = zxC - ys
    r[2, 1] = yzC + xs
    r[2, 2] = z*zC + ca

    return r


def r2aa(matrix):
    """Convert the rotation matrix into the axis-angle notation. Conversion equations

        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    """

    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = math.atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta

def euler(alpha, beta, gamma):
    """Compute  the z-y-z Euler angle convention rotation matrix.
                | -sin(alpha) * sin(gamma) + cos(alpha) * cos(beta) * cos(gamma) |
        mux  =  | -sin(alpha) * cos(gamma) - cos(alpha) * cos(beta) * sin(gamma) |.
                |                    cos(alpha) * sin(beta)                      |

                | cos(alpha) * sin(gamma) + sin(alpha) * cos(beta) * cos(gamma) |
        muy  =  | cos(alpha) * cos(gamma) - sin(alpha) * cos(beta) * sin(gamma) |.
                |                   sin(alpha) * sin(beta)                      |

                | -sin(beta) * cos(gamma) |
        muz  =  |  sin(beta) * sin(gamma) |.
                |        cos(beta)        |
    """

    # Trig.
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    sin_g = math.sin(gamma)

    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)

    r = np.zeros((3, 3))
    # The unit mux vector component of the rotation matrix.
    r[0, 0] = -sin_a * sin_g + cos_a * cos_b * cos_g
    r[1, 0] = -sin_a * cos_g - cos_a * cos_b * sin_g
    r[2, 0] = cos_a * sin_b

    # The unit muy vector component of the rotation matrix.
    r[0, 1] = cos_a * sin_g + sin_a * cos_b * cos_g
    r[1, 1] = cos_a * cos_g - sin_a * cos_b * sin_g
    r[2, 1] = sin_a * sin_b

    # The unit muz vector component of the rotation matrix.
    r[0, 2] = -sin_b * cos_g
    r[1, 2] = sin_b * sin_g
    r[2, 2] = cos_b

    return r

def r2euler(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

import numpy as np
import cv2
from pyquaternion import Quaternion
from cortex.geometry.rotation import reorthogonalize, rt2qt
import cortex.geometry.point as point

class SE3(object):
    def __init__(self, rotation=None, translation=None):
        ''' Constructor from a Quaternion and a 3D translation vector '''
        if rotation is None:
            rotation = Quaternion()

        if translation is None:
            translation = np.array([0, 0, 0])

        if not isinstance(rotation, Quaternion):
            # Assuming a numpy array is provided as input types
            if rotation.shape == (3, 3) or (4, 4):
                # Convert rotation/transformation matrix to a Quaternion
                self.rotation = Quaternion(matrix=rotation)
                # if transformation matrix is provided set the translation from the transformation matrix
                if rotation.shape == (4, 4):
                    translation = rotation[0:2, 3].ravel()
            else:
                raise ValueError('Unknown numpy matrix is provided. Either provide a rotation (3x3) or transformation matrix (4x4).')
        else:
            self.rotation = rotation

        self.translation = translation.ravel()

    @classmethod
    def from_array7(cls, v):
        return cls(rotation=Quaternion(np.array([v[3], v[0], v[1], v[2]])), translation=np.array([v[4], v[5], v[6]]))

    @classmethod
    def from_qt(cls, q, t):
        return cls(rotation=q, translation=t)

    @classmethod
    def from_rt(cls, r, t):
        return cls(rotation=Quaternion(matrix=r), translation=t)

    @classmethod
    def identity(cls):
        return cls(rotation=Quaternion(), translation=np.array([0, 0, 0]))

    def __repr__(self):
        return "SE3(%s, %s, %s, %s)" % tuple((repr(self.rotation), self.translation[0], self.translation[1], self.translation[2]))

    def __len__(self):
        return 7

    def __name__(self):
        return "SE3"

    @classmethod
    def from_transformation(cls, transformation):
        if transformation.shape[0] == 16:
            rotation = transformation.reshape(4,4)[0:3, 0:3]
            translation = transformation.reshape(4,4)[0:3, 3]
        else:
            rotation = transformation[0:3, 0:3]
            translation = transformation[0:3, 3]
        rotation = reorthogonalize(rotation)
        return cls(Quaternion(matrix=rotation), translation)

    def rotation_matrix(self):
        return self.rotation.rotation_matrix

    def transformation_matrix(self):
        transformation = np.zeros((4, 4))
        transformation[0:3, 0:3] = self.rotation.rotation_matrix
        transformation[0:3, 3] = self.translation.ravel()
        transformation[3, 3] = 1
        return transformation

    def matrix3x4(self):
        ''' Returns [R|T]'''
        return np.column_stack((self.rotation_matrix(), self.translation))

    def as_array(self):
        return np.array([self.rotation.elements[0], self.rotation.elements[1], self.rotation.elements[2], self.rotation.elements[3], self.translation[0], self.translation[1], self.translation[2]])

    def inverse(self):
        inv = self.rotation.inverse
        return SE3(inv, inv.rotate(-self.translation))

    def __mul__(self, other):
        if len(other) == 3:
            # transform the other point
            return self.rotation.rotate(other) + self.translation.T
        if len(other) == 4:
            # transform a homogeneous point
            x = np.ones(4)
            x[0:3] = self.rotation.rotate(other[0:3]) + self.translation
            return x
        elif len(other) == 7:
            # transform the other (SE3) transformation
            return SE3(self.rotation * other.rotation, self.translation + self.rotation.rotate(other.translation))
        else:
            raise ValueError('Cannot transfer an unknown object')

    def log(self):
        # todo implement the logarithmic map
        return np.zeros((3,1))

    def camera(self, K):
        ''' Returns K[R|T]'''
        return np.dot(K, self.matrix3x4())

    def perspective(self, K):
        w = 100
        h = 100
        r = self.rotation_matrix()
        Kinv = np.linalg.inv(K)
        # translation to origin, in world coordinates, = C = R.inv() * T
        # H1 = K * [I | -C] = K * R.inv() * K.inv()
        # H2 = [I| -K * C/Cz]
        # H = H1 * H2
        c = np.dot(r.T, self.translation)
        # Four sample points in the original image
        corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]], np.float32)
        scene = np.zeros((4, 2), np.float32)

        for i in range(4):
            p = point.homg(corners[i, :])
            # Direction of the ray that the corner images, in world coordinates
            lhat = np.squeeze(np.asarray(np.dot(r.T, np.dot(Kinv, p).T)))
            s = c[2] / lhat[2]
            # now we have the case that (s*lhat-u)[2]==0,
            # i.e. s is how far along the line of sight that we need
            # to move to get to the Z==0 plane.
            g = np.multiply(lhat - c, s)
            scene[i, :] = g[0:2].ravel()

        perspective_transformation = K * cv2.getPerspectiveTransform(corners, scene)
        return perspective_transformation

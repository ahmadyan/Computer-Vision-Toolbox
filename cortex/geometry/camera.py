import numpy as np
import cv2

def construct_K(fx=500.0, fy=500.0, cx=319.5, cy=239.5):
    """
    Create camera intrinsics from focal lengths and focal centers
    """
    K = np.eye(3)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    return K


def construct_D(k1=0, k2=0, k3=0, p1=0, p2=0):
    """
    Create camera distortion params
    """
    return np.float64([k1, k2, p1, p2, k3])


class CameraIntrinsic(object):
    def __init__(self, K, D=np.zeros(5, dtype=np.float64), shape=None):
        """
        K: Calibration matrix
        D: Distortion
        shape: Image Size (H,W,C): (480,640,3)
        """
        self.K = K
        self.D = D
        self.shape = np.int32(shape) if shape is not None else None

    def __repr__(self):
        return '\n' + '-' * 80 + '\nCameraIntrinsic:\n\t fx: {:3.2f}, fy: {:3.2f}, '\
            'cx: {:3.2f}, cy: {:3.2f}, \n\t shape: {:}, skew: {},\n\t D: {:}\n'.format(
                self.fx, self.fy, self.cx, self.cy, self.shape, self.skew,
                np.array_str(self.D, precision=2, suppress_small=True)) + '-' * 80 + '\n'


    @classmethod
    def simulate(cls, type=None):
        """
            https://zwiki.zillowgroup.net/display/zproddev/Camera+Calibration
            Simulate a 640x480 camera with 500 focal length

            Actual focal length in mm, should be converted to pixel
            focal (mm) / sensor-width * image-width
            1/3″ (4.80 x 3.60 mm)
            focal = 4.15
        """
        if type == "iphone6+":
            return cls.from_calib_params(1660., 1660., 959.5, 539.5, shape=(1080, 1920))
        elif type == "iphone7+":
            return cls.from_calib_params(1560., 1560., 320., 240., shape=(1080, 1920))
        else:
            return cls.from_calib_params(500., 500., 320., 240., shape=(480, 640))

    @classmethod
    def from_calib_params(cls, fx, fy, cx, cy, k1=0, k2=0, k3=0, p1=0, p2=0, shape=None):
        return cls(construct_K(fx, fy, cx, cy),
                   D=construct_D(k1, k2, k3, p1, p2), shape=shape)

    @classmethod
    def from_calib_params_fov(cls, fov, cx, cy, D=np.zeros(5, dtype=np.float64), shape=None):
        return cls(construct_K(cx / np.tan(fov), cy / np.tan(fov), cx, cy), D=D, shape=shape)

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    @property
    def skew(self):
        return self.K[0, 1]

    @property
    def k1(self):
        return self.D[0]

    @property
    def k2(self):
        return self.D[1]

    @property
    def k3(self):
        return self.D[4]

    @property
    def p1(self):
        return self.D[2]

    @property
    def p2(self):
        return self.D[3]

    @property
    def fov(self):
        """
        Returns the field of view for each axis
        """
        return np.float32([np.arctan(self.shape[1] * 0.5 / self.fx),
                           np.arctan(self.shape[0] * 0.5 / self.fy)]) * 2.0

    def scaled(self, scale):
        """
        Returns the scaled intrinsics of the camera, that results
        from resizing the image by the appropriate scale parameter
        """
        shape = np.int32(
            self.shape * scale) if self.shape is not None else None
        return CameraIntrinsic.from_calib_params(self.fx * scale, self.fy * scale,
                                                 self.cx * scale, self.cy * scale,
                                                 k1=self.k1, k2=self.k2, k3=self.k3,
                                                 p1=self.p1, p2=self.p2,
                                                 shape=shape)

    def in_view(self, x):
        """ Only return points within-image bounds """
        return np.where(np.bitwise_and(np.bitwise_and(x[:, 0] >= 0, x[:, 0] < self.shape[1]),
                                       np.bitwise_and(x[:, 1] >= 0, x[:, 1] < self.shape[0])))[0]

    def ray(self, pts, undistort=True, rotate=False, normalize=False):
        """
        Returns the ray corresponding to the points.
        Optionally undistort (defaults to true), and
        rotate ray to the camera's viewpoint
        """
        upts = self.undistort_points(pts) if undistort else pts
        ret = unproject_points(
            np.hstack([(colvec(upts[:, 0]) - self.cx) / self.fx,
                       (colvec(upts[:, 1]) - self.cy) / self.fy])
        )

        if rotate:
            ret = self.extrinsics.rotate_vec(ret)

        if normalize:
            ret = ret / np.linalg.norm(ret, axis=1)[:, np.newaxis]

        return ret

    def reconstruct(self, xyZ, undistort=True):
        """
        Reproject to 3D with calib params
        """
        Z = colvec(xyZ[:, 2])
        return self.ray(xyZ[:, :2], undistort=undistort) * Z

    def undistort(self, im):
        return undistort_image(im, self.K, self.D)

    def undistort_points(self, pts):
        """
        Undistort image points using camera matrix, and distortion coeffs
        Have to provide P matrix for appropriate scaling
        http://code.opencv.org/issues/1964#note-2
        """
        out = cv2.undistortPoints(
            pts.reshape(-1, 1, 2).astype(np.float32), self.K, self.D, P=self.K)
        return out.reshape(-1, 2)

    def undistort_debug(self, im=None):
        if im is None:
            im = np.zeros(shape=self.shape, dtype=np.uint8)
        im[::20, :] = 128
        im[:, ::20] = 128
        return self.undistort(im)

    # def save(self, filename):
    #     try:
    #         height, width = self.shape[:2]
    #     except:
    #         height, width = '', ''
    #     AttrDict(
    #         fx=float(self.fx), fy=float(self.fy),
    #         cx=float(self.cx), cy=float(self.cy),
    #         k1=float(self.D[0]), k2=float(self.D[1]), k3=float(self.D[4]), p1=float(self.D[2]), p2=float(self.D[3]),
    #         width=int(width), height=int(height)
    #     ).save_yaml(filename)
    #
    # @classmethod
    # def load(cls, filename):
    #     db = AttrDict.load_yaml(filename)
    #     shape = np.int32([db.width, db.height]) if hasattr(
    #         db, 'width') and hasattr(db, 'height') else None
    #     return cls.from_calib_params(db.fx, db.fy, db.cx, db.cy,
    #                                  k1=db.k1, k2=db.k2, k3=db.k3, p1=db.p1, p2=db.p2, shape=shape)





def pixel2metric(x, K):
    '''
    Convert pixel measurements to metric measurements
    using (pixel - principal) / focal
    :param x: 2D point (x,y) in pixel space
    :param K: 3x3 camera intrinsic matrix
    :return:  2D numpy array (x,y) in metric space
    '''
    y = np.zeros(2)
    y[0] = (x[0] - K[0,2]) / K[0,0]
    y[1] = (x[1] - K[1,2]) / K[1,1]
    return y


def metric2pixel(x, K):
    '''
        Convert metric measurements to pixel measurements
        using focal * metric + principal
        :param x: 2D point (x,y) in metric space
        :param K: 3x3 camera intrinsic matrix
        :return:  2D numpy array (x,y) in pixel space
    '''
    y = np.zeros(2)
    y[0] = (K[0,2] * x[0]) + K[0,0]
    y[1] = (K[1,2] * x[1]) + K[1,1]
    return x

def visibility_test(camera, pts_w, zmin=0, zmax=100):
    """
    Check if points are visible given fov of camera.
    This method checks for both horizontal and vertical
    fov.
    camera: type Camera
    """
    # Transform points in to camera's reference
    # Camera: p_cw
    pts_c = camera.c2w(pts_w.reshape(-1, 3))

    # Determine look-at vector, and check angle
    # subtended with camera's z-vector (3rd column)
    z = pts_c[:, 2]
    v = pts_c / np.linalg.norm(pts_c, axis=1).reshape(-1, 1)
    hangle, vangle = np.arctan2(
        v[:, 0], v[:, 2]), np.arctan2(-v[:, 1], v[:, 2])

    # Provides inds mask for all points that are within fov
    return np.bitwise_and.reduce([np.fabs(hangle) < camera.fov[0] * 0.5,
                                  np.fabs(vangle) < camera.fov[1] * 0.5,
                                  z >= zmin, z <= zmax])


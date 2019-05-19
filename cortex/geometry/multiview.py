import numpy as np
import cortex.matrix.utils as matrix


def homg(x):
  """Homogenize vector """
  return np.append(x, 1)


def project(x):
  """De-homogenize vector """
  return x[:-1] / float(x[-1])

def unproject_points(pts):
  assert (pts.ndim == 2 and pts.shape[1] == 2)
  return np.hstack([pts, matrix.colvec(np.ones(len(pts)))])


def project_points(pts):
  z = matrix.colvec(pts[:, 2])
  return pts[:, :2] / z


def inverse_transform(T):
  R = T[:3, :3].T
  p = T[:3, 3]
  T_inv = np.identity(4)
  T_inv[:3, :3] = R
  T_inv[:3, 3] = np.dot(-R, p)
  return T_inv

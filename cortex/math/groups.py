import numpy as np


def Hat(vec):
  """ Skew matrix A such that a x v = Av for any v. """
  if vec.shape[0] == 3: # skew from vec
    return np.array([[0,-vec[2],vec[1]],[vec[2],0,-vec[0]],[-vec[1],vec[0],0]])
  elif vec.shape[0] == 6:
    vechat = np.zeros((4,4))
    vechat[:3,:3] = Hat(vec[3:])
    vechat[:3,3] = vec[:3]
    return vechat
  else:
    raise ValueError("Invalid vector length for hat operator\n")


def VecFromSkew(r):
  return np.array([r[2,1],r[0,2],r[1,0]])


def CurlyHat(vec):
  veccurlyhat = np.zeros((6,6))
  veccurlyhat[:3,:3] = Hat(vec[3:])
  veccurlyhat[:3,3:] = Hat(vec[:3])
  veccurlyhat[3:,3:] = Hat(vec[3:])
  return veccurlyhat

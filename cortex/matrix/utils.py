import numpy as np


def colvec(vec):
  """Convert to column vector """
  return vec.reshape(-1, 1)


def rowvec(vec):
  """Convert to row vector """
  return vec.reshape(1, -1)


def flatten(x):
  return np.squeeze(np.asarray(x))

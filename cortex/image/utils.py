"""General utility functions for images, notably im2col and col2im

This file contains some general utility functions related to images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def im2col(img, block_size, step=1):
  """
    Rearrange image blocks into columns
    port of Matlab's im2col (sliding)
    https://www.mathworks.com/help/images/ref/im2col.html
    @param img: grayscale image
    @result each patch is output[:, i]
    """
  m, n = img.shape
  s0, s1 = img.strides
  nrows = m - block_size[0] + 1
  ncols = n - block_size[1] + 1
  shp = block_size[0], block_size[1], nrows, ncols
  strd = s0, s1, s0, s1

  out_view = np.lib.stride_tricks.as_strided(img, shape=shp, strides=strd)
  return out_view.reshape(block_size[0] * block_size[1], -1)[:, ::step]


def col2im(B, block_size, image_size):
  """
    Rearrange matrix columns into blocks
    """
  m, n = block_size
  mm, nn = image_size
  print(m, n, mm, nn)
  return B.reshape(nn - n + 1, mm - m + 1).T
  #B.reshape(nn-n+1,-1).T
  #B.reshape(mm-m+1,nn-n+1,order='F')


def im2corners(image):
  height = image.shape[0]
  width = image.shape[1]
  obj_corners = np.empty((4, 1, 2), dtype=np.float32)
  obj_corners[0, 0, 0] = 0
  obj_corners[0, 0, 1] = 0
  obj_corners[1, 0, 0] = width
  obj_corners[1, 0, 1] = 0
  obj_corners[2, 0, 0] = width
  obj_corners[2, 0, 1] = height
  obj_corners[3, 0, 0] = 0
  obj_corners[3, 0, 1] = height
  return obj_corners


def undistort_image(im, K, D):
  """
    Optionally: newcamera, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D,
    (W,H), 0)
    """
  H, W = im.shape[:2]
  Kprime, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 1, (W, H))
  return cv2.undistort(im, K, D, None, K)

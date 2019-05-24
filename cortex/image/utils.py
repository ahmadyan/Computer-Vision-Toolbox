"""General utility functions for images, notably im2col and col2im

This file contains some general utility functions related to images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from PIL import Image


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


def iou(box1, box2):
  """Computes Intersection over Union value for 2 bounding boxes.

  Args:
    box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    box2: same as box1

  Returns:
    IoU
  """
  b1_x0, b1_y0, b1_x1, b1_y1 = box1
  b2_x0, b2_y0, b2_x1, b2_y1 = box2

  int_x0 = max(b1_x0, b2_x0)
  int_y0 = max(b1_y0, b2_y0)
  int_x1 = min(b1_x1, b2_x1)
  int_y1 = min(b1_y1, b2_y1)

  int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

  b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
  b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

  # we add small epsilon of 1e-05 to avoid division by 0
  iou = int_area / (b1_area + b2_area - int_area + 1e-05)
  return iou


def resize_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

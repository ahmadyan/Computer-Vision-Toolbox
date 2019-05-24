# Interface for Pillow's ImageFont 
# https://pillow.readthedocs.io/en/3.0.x/reference/ImageFont.html

import os
import numpy as np
from PIL import ImageFont


def load_font(image_height = 416):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  font_path = os.path.join(dir_path, 'font', 'FiraMono-Medium.otf'); 
  size = np.floor(3e-2 * image_height + 0.5).astype('int32')
  return ImageFont.truetype(font=font_path, size=size)

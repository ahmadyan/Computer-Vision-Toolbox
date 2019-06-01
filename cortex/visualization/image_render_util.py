import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import NoNorm
import cv2


def save_image(data, filename, size=(1, 1), dpi=600):
    if len(data.shape) == 2:
        color_map = 'gray'
    else:
        color_map = 'hot'
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('gray')
    ax.imshow(data, aspect='equal', norm=NoNorm())
    plt.savefig(filename, dpi=dpi)


def render(image):
    if len(image.shape) == 2:
        color_map = 'gray'
    else:
        color_map = 'hot'
    fig = plt.figure()
    #fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap(color_map)
    ax.imshow(image, aspect='equal', norm=NoNorm())
    plt.show()


def draw_box(image, corners, height=None, width=None, color=None):
    if height is None:
        height = image.shape[0]
    if width is None:
        width = image.shape[1]
    if color is None:
        color = (0, 255, 0)
    cv2.line(image, (int(corners[0, 0, 0] + width), int(corners[0, 0, 1])), \
             (int(corners[1, 0, 0] + width), int(corners[1, 0, 1])), color, 4)
    cv2.line(image, (int(corners[1, 0, 0] + width), int(corners[1, 0, 1])), \
             (int(corners[2, 0, 0] + width), int(corners[2, 0, 1])), color, 4)
    cv2.line(image, (int(corners[2, 0, 0] + width), int(corners[2, 0, 1])), \
             (int(corners[3, 0, 0] + width), int(corners[3, 0, 1])), color, 4)
    cv2.line(image, (int(corners[3, 0, 0] + width), int(corners[3, 0, 1])), \
             (int(corners[0, 0, 0] + width), int(corners[0, 0, 1])), color, 4)
    return image


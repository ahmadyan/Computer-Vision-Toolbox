import math


def GRIC_score(e, d, k, r):
    '''
    Torr's Geometric Motion Segmentation and Model Selection

    For homography: GRIC(He, 2, 8, 4)
    For fundamental matrix: GRIC(Fe, 3, 7, 4)

    :param e: is the error residuals
    :param d: is the number of dimensions modeled (d = 3 for a fundamental matrix or 2 for a homography)
    :param k: is the number of degrees of freedom in the model (k = 7 for a fundamental matrix or 8 for a homography)
    :param r: is the dimension of the data (r = 4 for 2D correspondences between two frames)
    :return: the gric score of the model (lower is better)
    '''

    n = len(e)
    lambda1 = math.log(r)
    lambda2 = math.log(r * n)

    # lambda3 limits the residual error, and this paper
    # http://elvera.nue.tu-berlin.de/files/0990Knorr2006.pdf
    # suggests using lambda3 of 2
    # same value is used in Torr's Problem of degeneracy in structure
    # and motion recovery from uncalibrated image sequences
    # http://www.robots.ox.ac.uk/~vgg/publications/papers/torr99.ps.gz
    lambda3 = 2.0

    # Variance of tracker position. Physically, this is typically about 0.1px,
    # and when squared becomes 0.01 px^2.
    sigma2 = 0.01

    # Calculate the GRIC score.
    gric = 0.
    for i in range(n):
        gric += min(e[i] * e[i] / sigma2, lambda3 * (r - d))

    gric += lambda1 * d * n
    gric += lambda2 * k
    return gric

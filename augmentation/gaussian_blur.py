import numpy as np
from scipy.ndimage.filters import gaussian_filter

class GaussianBlur(object):
    """ apply gaussian blur to numpy array
    """

    def __init__(self, signma=1):
        self.sigma = signma

    def __call__(self, np_array):
        return gaussian_filter(np_array, [self.sigma, self.sigma, 0])
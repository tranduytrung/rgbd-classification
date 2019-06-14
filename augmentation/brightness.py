import numpy as np
import torch

class Brightness(object):
    """ apply translate value by uniform random
    """

    def __init__(self, minmax=(-0.8, 0.8)):
        self.minmax = minmax

    def __call__(self, tensor):
        add_value = torch.rand(1)*(self.minmax[1] - self.minmax[0]) + self.minmax[0]
        return tensor + add_value

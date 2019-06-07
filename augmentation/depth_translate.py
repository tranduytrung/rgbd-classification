import numpy as np
import torch

class DepthTranslate(object):
    """ apply translate depth value by uniform random
    """

    def __init__(self, minmax=(-0.8, 0.8), exclude_gt=0.98):
        self.minmax = minmax
        self.exclude_gt = exclude_gt

    def __call__(self, tensor):
        add_value = torch.rand(1)*(self.minmax[1] - self.minmax[0]) + self.minmax[0]
        if self.exclude_gt is None:
            return tensor + add_value
        
        tensor[tensor < self.exclude_gt] += add_value
        return tensor

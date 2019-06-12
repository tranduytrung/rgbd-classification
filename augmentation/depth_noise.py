import numpy as np
import torch


def gaussian_kernel(size=3, sigma=1):
    """Returns a 2D Gaussian kernel."""

    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.e ** (
        (-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    return kernel / kernel[int(size//2), int(size//2)]


class DepthNoise(object):
    """ apply gaussian noise to PIL image

    Args:
        size (int): the size of input image
        sigma (float): the sigma of gaussian distribution
        pivot (float): the pivot point taken as prob 1 calculating from the corner (0.0) to center (1.0).
    """

    def __init__(self, size=224, sigma=112, strength=1.0, minmax=(-1.0, 1.0)):
        k = gaussian_kernel(size, sigma)
        self.p = torch.tensor(k*strength, dtype=torch.float)
        self.size = size
        self.minmax = minmax

    def __call__(self, tensor):
        minmax = self.minmax
        mask = torch.rand_like(tensor) > self.p
        tensor[mask] = torch.randn(torch.sum(mask)) * (minmax[1] - minmax[0]) + minmax[0]

        return tensor

class DepthUniformNoise(object):
    """ apply uniform noise to PIL image

    Args:
        p (float): prob that a pixel turn random
    """

    def __init__(self, p=0.1, minmax=(-1.0, 1.0)):
        self.p = p
        self.minmax = minmax

    def __call__(self, tensor):
        minmax = self.minmax
        mask = torch.rand_like(tensor) < self.p
        if isinstance(minmax, (int, float)):
            tensor[mask] = minmax
        else:
            tensor[mask] = torch.randn(torch.sum(mask)) * (minmax[1] - minmax[0]) + minmax[0]

        return tensor
import numpy as np
import torch


def gaussian_kernel(size=3, sigma=1):
    """Returns a 2D Gaussian kernel."""

    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.e ** (
        (-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    return kernel / kernel.sum()


class DepthNoise(object):
    """ apply gaussian noise to PIL image

    Args:
        size (int): the size of input image
        sigma (float): the sigma of gaussian distribution
        pivot (float): the pivot point taken as prob 1 calculating from the corner (0.0) to center (1.0).
    """

    def __init__(self, size=224, sigma=112, pivot=1.0):
        k = gaussian_kernel(size, sigma)
        self.p = torch.tensor(
            k/k[int(pivot*size)//2, int(pivot*size)//2], dtype=torch.float)
        self.size = size

    def __call__(self, tensor):
        mask = torch.rand_like(tensor) > self.p
        tensor[mask] = torch.randn(torch.sum(mask))

        return tensor

import torch

class Clamp(object):
    """ clamp the value
    """

    def __init__(self, clamp=(-1, 1)):
        self.clamp = clamp

    def __call__(self, tensor):
        return torch.clamp(tensor, *self.clamp)
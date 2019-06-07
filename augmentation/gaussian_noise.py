import torch

class GaussianNoise(object):
    """ apply gaussian noise to PIL image
    """

    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        added_noise = torch.randn_like(tensor) * self.std + self.mean
        noised_tensor = tensor + added_noise
        return noised_tensor
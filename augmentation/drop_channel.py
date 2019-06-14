import torch

class DropChannel(object):
    """ randomly drop channels of tensor
    """

    def __init__(self, channels=None, fill_value=0):
        self.channels = channels
        self.fill_value = fill_value

    def __call__(self, tensor: torch.Tensor):
        channels = self.channels
        if channels is None:
            channels = range(tensor.shape[0])

        drop_index = torch.randint(0, len(channels) + 1, [1]).item()
        # no drop
        if drop_index == len(channels):
            return tensor

        # drop one
        selected_channel = channels[drop_index]
        if hasattr(self.fill_value, '__getitem__'):
            tensor[selected_channel, ...] = self.fill_value[drop_index]
        else:
            tensor[selected_channel, ...] = self.fill_value

        return tensor
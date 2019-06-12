import random
import math
from .functional import crop, resize

class CenterCrop(object):
    """ crop at center of image """
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(in_size, out_size):
        h = min(in_size[0], out_size[0])
        w = min(in_size[1], out_size[1])
        
        i = (in_size[0] - h) // 2
        j = (in_size[1] - w) // 2

        return i, j, w, h

    def __call__(self, nparr):
        in_size = nparr.shape[:2]
        y, x, h, w = CenterCrop.get_params(in_size, self.size)
        cropped = crop(nparr, y, x, h, w)
        resized = resize(cropped, self.size)
        return resized


class CropAndResize(object):
    """ crop and resize 2D tensor
    """

    def __init__(self, size, scale=(0.25, 1.0), ratio=(3./4., 4./3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(size, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
         Args:
             size (tuple): input size (h, w)
             scale (tuple): range of size of the origin size cropped
             ratio (tuple): range of aspect ratio of the origin aspect ratio cropped (w, h)
         Returns:
             tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                 sized crop.
         """
        area = size[0]*size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[1] and h <= size[0]:
                i = random.randint(0, size[0] - h)
                j = random.randint(0, size[1] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = size[1] / size[0]
        if (in_ratio < min(ratio)):
            w = size[1]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = size[0]
            w = h * max(ratio)
        else:  # whole image
            w = size[1]
            h = size[0]
        i = (size[0] - h) // 2
        j = (size[1] - w) // 2
        return i, j, h, w

    def __call__(self, nparr):
        y, x, h, w = CropAndResize.get_params(nparr.shape[:2], self.scale, self.ratio)
        cropped = crop(nparr, y, x, h, w)
        resized = resize(cropped, self.size)
        return resized